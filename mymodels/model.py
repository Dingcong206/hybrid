import torch
import torch.nn as nn
from mamba_ssm import Mamba


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
    def __init__(self, d_model, nhead=8, dropout=0.3, mamba_blocks=3):
        super().__init__()
        # 局部卷积提取
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 前置 Mamba 组
        self.mambas_pre = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout) for _ in range(mamba_blocks)
        ])

        # 中间注意力层
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 后置 Mamba 组
        self.mambas_post = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout) for _ in range(mamba_blocks)
        ])

        # 门控残差
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x, mask=None):
        res = x
        # 1. 局部特征
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # 2. 前置双向 Mamba
        for blk in self.mambas_pre:
            x = blk(x)

        # 3. 全局注意力
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask, need_weights=False)
        x = x + x_a

        # 4. 后置双向 Mamba
        for blk in self.mambas_post:
            x = blk(x)

        # 5. 门控输出
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
def build_model(in_dim=1024, d_model=512, n_layers=6, nhead=8, dropout=0.3):
    model = SSA_Model_HeARTokens(
        in_dim=in_dim, d_model=d_model, n_layers=n_layers, nhead=nhead, dropout=dropout
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SSA Model Initialized. Parameters: {params:,}")
    return model