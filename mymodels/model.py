import torch
import torch.nn as nn
from mamba_ssm import Mamba

from mymodels.wrapper import SSA4ClassWrapper

# =========================
# 1. 位置编码函数
# =========================
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


# =========================
# 2. 基础 BiMamba 块
# =========================
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
        h = self.ln1(x)
        # 双向 Mamba 合并
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        return x + self.mlp(self.ln2(x))


# =========================
# 3. SSA 层 (单层定义)
# =========================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # 1. 局部特征提取
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 2. 前置 4 个 BiMamba
        self.mambas_pre = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout) for _ in range(4)
        ])

        # 3. 中间 2 个 Attention 层 (带残差和归一化)
        self.attn1_ln = nn.LayerNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.attn2_ln = nn.LayerNorm(d_model)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 4. 后置 4 个 BiMamba
        self.mambas_post = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout) for _ in range(4)
        ])

        # 5. 门控残差
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        res = x

        # --- Local Conv ---
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # --- Pre Mambas (4 layers) ---
        for blk in self.mambas_pre:
            x = blk(x)

        # --- Dual Attention (2 layers) ---
        # 第一层 Attention
        x_n1 = self.attn1_ln(x)
        x_a1, _ = self.attn1(x_n1, x_n1, x_n1, key_padding_mask=mask)
        x = x + x_a1

        # 第二层 Attention
        x_n2 = self.attn2_ln(x)
        x_a2, _ = self.attn2(x_n2, x_n2, x_n2, key_padding_mask=mask)
        x = x + x_a2

        # --- Post Mambas (4 layers) ---
        for blk in self.mambas_post:
            x = blk(x)

        # --- Gated Residual ---
        g = self.gate(x.mean(dim=1, keepdim=True))
        return res + g * x
# =========================
# 4. 完整的 SSA 模型
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(self, in_dim=1024, d_model=512, n_layers=6, nhead=8, dropout=0.3, max_len=2000):
        super().__init__()

        # 输入投影：将 HeAR 的 1024 映射到 d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 位置编码 buffer (不计入梯度，随模型移动)
        pe = sinusoidal_positional_encoding(max_len, d_model, device='cpu')
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        self.pos_drop = nn.Dropout(dropout)

        # 堆叠 SSA 层
        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 注意力池化 (Attention Pooling)
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        self.classifier = nn.Linear(d_model, 1)  # 文件级分类 logit
        self.token_head = nn.Linear(d_model, 1)  # Token 级分类 logit

    def forward(self, x, mask=None):
        """
        x: (B, T, in_dim)
        mask: (B, T) True 代表 padding 位置
        """
        # 1. 投影与位置编码
        x = self.input_proj(x)
        B, T, D = x.shape
        x = x + self.pe[:, :T, :]
        x = self.pos_drop(x)

        # 2. 经过多层 SSA
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        # 3. 池化得到全局特征 (File-level)
        attn_scores = self.attention_net(x)  # (B, T, 1)
        if mask is not None:
            # 屏蔽 padding 部分的权重
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)

        attn_w = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        file_feature = torch.sum(attn_w * x, dim=1)  # (B, d_model)

        # 4. 输出层
        file_logit = self.classifier(file_feature).squeeze(-1)  # (B,)
        token_logits = self.token_head(x).squeeze(-1)  # (B, T)

        return file_logit, token_logits


# =========================
# 5. 构建函数
# =========================
def build_model(in_dim=1024, d_model=512, n_layers=6, nhead=8, dropout=0.3, num_classes=4):
    base = SSA_Model_HeARTokens(
        in_dim=in_dim, d_model=d_model, n_layers=n_layers, nhead=nhead, dropout=dropout
    )


    model = SSA4ClassWrapper(
        base_model=base,
        feat_dim=None,  # 先不指定，让 wrapper 自动推断
        num_classes=num_classes,
        dropout=dropout
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SSA 4-Class Model Initialized. Parameters: {params:,}")
    return model
