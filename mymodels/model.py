import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================
# 1. 辅助模块：位置编码
# =====================================================
def sinusoidal_positional_encoding(seq_len: int, dim: int, device):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (
                -torch.log(torch.tensor(10000.0, device=device)) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =====================================================
# 2. 核心模块：4倍下采样器 (32768 -> 8192)
# =====================================================
class Downsampler(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 路径1：卷积学习下采样
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4)
        # 路径2：最大池化保留异常突发音（如爆裂音）
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.fuse = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x_raw = x.transpose(1, 2)
        x1 = self.conv(x_raw)
        x2 = self.maxpool(x_raw)

        x_fused = torch.cat([x1, x2], dim=1).transpose(1, 2)
        return self.norm(self.fuse(x_fused))


# =====================================================
# 3. 核心模块：双向 Mamba 块
# =====================================================
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
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.ln1(x)
        # 并行计算双向 Mamba 增加上下文理解
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        return x + self.mlp(self.ln2(x))


# =====================================================
# 4. 核心模块：SSA 层 (Conv + Mamba + Attention)
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.mamba = BiMambaBlock(d_model, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x, mask=None):
        res = x
        # 1. 局部特征提取
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        # 2. 序列建模
        x = self.mamba(x)
        # 3. 全局注意力
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        x = x + x_a
        # 4. 门控残差
        g = self.gate(x.mean(dim=1, keepdim=True))
        return res + g * x


# =====================================================
# 5. 最终模型定义
# =====================================================
class SSA_Model_8k(nn.Module):
    def __init__(self, in_dim=256, d_model=256, n_layers=4, nhead=8):
        super().__init__()
        self.downsampler = Downsampler(d_model=in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.patch_head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        Input x: (B, 32768, 256)
        """
        # 1. 4倍下采样 -> (B, 8192, 256)
        x = self.downsampler(x)

        # 2. 映射与位置编码
        x = self.input_proj(x)
        B, T, D = x.shape
        pos = sinusoidal_positional_encoding(T, D, x.device).unsqueeze(0)
        x = x + pos

        # 3. 同步下采样 Mask
        if mask is not None:
            mask = mask[:, ::4]

        # 4. SSA 主干
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        # 5. 输出：用于多实例学习 (MIL)
        logits = self.patch_head(x).squeeze(-1)  # (B, 8192)
        file_logit, _ = torch.max(logits, dim=1)

        return file_logit, logits


# =====================================================
# 6. 模型构建函数
# =====================================================
def build_model(in_dim=256, d_model=256, n_layers=4, nhead=8):
    model = SSA_Model_8k(in_dim, d_model, n_layers, nhead)
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SSA Model Initialized. Parameters: {params:,}")
    return model