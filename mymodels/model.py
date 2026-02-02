import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================
# 1) 位置编码
# =====================================================
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


# =====================================================
# 2) 可配置下采样器：factor=4 / 8 / 16 ...
#    这里我们用 factor=4，叠两次得到 16x
# =====================================================
class Downsampler(nn.Module):
    def __init__(self, d_model: int, factor: int = 4):
        super().__init__()
        self.factor = factor

        # 路径1：卷积下采样
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=factor, stride=factor)

        # 路径2：最大池化下采样（保留突发峰值）
        self.maxpool = nn.MaxPool1d(kernel_size=factor, stride=factor)

        # 融合
        self.fuse = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T/factor, D)
        """
        x_raw = x.transpose(1, 2)  # (B, D, T)
        x1 = self.conv(x_raw)      # (B, D, T/f)
        x2 = self.maxpool(x_raw)   # (B, D, T/f)

        x_fused = torch.cat([x1, x2], dim=1).transpose(1, 2)  # (B, T/f, 2D)
        x_fused = self.fuse(x_fused)                          # (B, T/f, D)
        return self.norm(x_fused)


# =====================================================
# 3) 双向 Mamba
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
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        return x + self.mlp(self.ln2(x))


# =====================================================
# 4) SSA 层
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.mamba = BiMambaBlock(d_model, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x, mask=None):
        """
        x: (B, T, D)
        mask: (B, T) bool, True=PAD (key_padding_mask 的语义)
        """
        res = x

        # local
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # sequence modeling
        x = self.mamba(x)

        # global attention
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask, need_weights=False)
        x = x + x_a

        # gated residual
        g = self.gate(x.mean(dim=1, keepdim=True))
        return res + g * x


# =====================================================
# 5) 最终模型：两次×4 下采样 -> 2048
# =====================================================
class SSA_Model_2k(nn.Module):
    def __init__(self, in_dim=256, d_model=256, n_layers=4, nhead=8,topk_ratio=0.1):
        super().__init__()
        self.topk_ratio = topk_ratio
        # 两次 factor=4：32768 -> 8192 -> 2048
        self.down1 = Downsampler(d_model=in_dim, factor=4)
        self.down2 = Downsampler(d_model=in_dim, factor=4)

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.patch_head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        x: (B, 32768, 256)
        mask: (B, 32768) bool (可选)
        return:
          file_logit: (B,)
          logits: (B, 2048)
        """

        # ---- 下采样 1：-> 8192
        x = self.down1(x)
        if mask is not None:
            mask = mask[:, ::4]

        # ---- 下采样 2：-> 2048
        x = self.down2(x)
        if mask is not None:
            mask = mask[:, ::4]   # 再 /4，总共 /16

        # ---- proj + pos
        x = self.input_proj(x)    # (B, 2048, d_model)
        B, T, D = x.shape
        pos = sinusoidal_positional_encoding(T, D, x.device).unsqueeze(0)
        x = x + pos

        # ---- backbone
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)


        # ---- 【改进点】Top-K 聚合策略 ----
        # 计算需要取多少个点 (例如 2048 * 0.1 = 204个点)
        k = int(T * self.topk_ratio)

        # 对每个样本的 logits 进行排序，取最大的前 k 个
        topk_logits, _ = torch.topk(logits, k, dim=1)

        # 取均值作为文件级别的最终 Logit
        file_logit = torch.mean(topk_logits, dim=1)  # (B,)

        return file_logit, logits


# =====================================================
# 6) build_model
# =====================================================
def build_model(in_dim=256, d_model=256, n_layers=4, nhead=8):
    model = SSA_Model_2k(in_dim=in_dim, d_model=d_model, n_layers=n_layers, nhead=nhead)
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SSA Model (2k) Initialized. Parameters: {params:,}")
    return model
