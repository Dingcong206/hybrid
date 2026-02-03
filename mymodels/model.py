import torch
import torch.nn as nn
from mamba_ssm import Mamba


def sinusoidal_positional_encoding(seq_len: int, dim: int, device):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, D)
        h = self.ln1(x)
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        return x + self.mlp(self.ln2(x))


class SSA_Layer(nn.Module):
    """
    一个 layer 的结构变成：
    local conv
      -> BiMamba x3
      -> Self-Attention x1
      -> BiMamba x3
      -> gated residual
    """

    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # 1. 增强型局部提取 (Depthwise Separable + Dilated)
        # 增加空洞卷积以在不增加参数量的情况下扩大感受野
        self.local_extract = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=4, dilation=2, groups=d_model),
            nn.BatchNorm1d(d_model),
        )

        # 2. 并行混合层 (Hybrid Mamba & Attention)
        # 让 Mamba 处理时序扫描，Attention 处理全局对齐
        self.mamba_branch = BiMambaBlock(d_model, dropout=dropout)

        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 3. 动态融合门控 (Cross-Gate)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # 4. 前馈网络 (FFN) 升级为 SwiGLU 风格 (Transformer 常用优化)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),  # Swish 激活
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        shortcut = x

        # --- A. 局部卷积分支 ---
        x_conv = self.local_extract(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        # --- B. 并行双路分支 ---
        # 支路1: Mamba (时序敏感)
        x_mamba = self.mamba_branch(x)

        # 支路2: Attention (全局对齐)
        x_n = self.attn_ln(x)
        x_attn, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)

        # 动态融合: 根据特征内容决定两者的权重
        gate = self.fusion_gate(torch.cat([x_mamba, x_attn], dim=-1))
        x = (1 - gate) * x_mamba + gate * x_attn
        x = x + shortcut  # 残差连接

        # --- C. FFN 增强层 ---
        x = x + self.ffn(self.norm_ffn(x))

        return x



class SSA_Model_HeARTokens(nn.Module):
    """
    输入：HeAR tokens (B, T, 1024)，你现在 T≈200（也可变）
    输出：file_logit (B,), token_logits (B, T)
    """
    def __init__(self, in_dim=1024, d_model=512, n_layers=6, nhead=8, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Attention Pooling (learnable)
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        self.classifier = nn.Linear(d_model, 1)
        self.token_head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        x: (B, T, 1024)
        mask: (B, T) True=padding
        """
        x = self.input_proj(x)  # (B, T, d_model)

        B, T, D = x.shape
        pos = sinusoidal_positional_encoding(T, D, x.device).unsqueeze(0)  # (1,T,D)
        x = x + pos

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)  # (B, T, d_model)

        # Attention pooling -> file_feature
        attn_scores = self.attention_net(x)  # (B, T, 1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)

        attn_w = torch.softmax(attn_scores, dim=1)        # (B, T, 1)
        file_feature = torch.sum(attn_w * x, dim=1)       # (B, d_model)

        file_logit = self.classifier(file_feature).squeeze(-1)  # (B,)
        token_logits = self.token_head(x).squeeze(-1)           # (B, T)

        return file_logit, token_logits


def build_model(in_dim=1024, d_model=512, n_layers=4, nhead=8, dropout=0.3):
    model = SSA_Model_HeARTokens(
        in_dim=in_dim, d_model=d_model, n_layers=n_layers, nhead=nhead, dropout=dropout
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SSA Model for HeAR tokens Initialized. Parameters: {params:,}")
    return model
