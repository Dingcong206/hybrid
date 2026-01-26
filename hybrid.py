import torch
import torch.nn as nn

# 尝试导入 Mamba 内核，如果没装，会报错提示
try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm)")


# =====================================================
# 第一部分：声学条带卷积 (Inspired by HeAR)
# 将 (128, 1024) 的频谱切成一个个“时间窄、频率宽”的条带
# =====================================================
class AcousticStripStem(nn.Module):
    def __init__(self, freq_bins=128, patch_time=4, embed_dim=192):
        super().__init__()
        # 卷积核高度=128(全频率), 宽度=patch_time(窄时间)
        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=(freq_bins, patch_time),
            stride=(freq_bins, patch_time)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 1, 128, 1024]
        x = self.proj(x)  # -> [B, embed_dim, 1, L]
        x = x.flatten(2).transpose(1, 2)  # -> [B, L, embed_dim]
        return self.norm(x)


# =====================================================
# 第二部分：混合 Mamba-Attention 块 (Inspired by Vim & Hybrid Logic)
# 结合了 Mamba 的序列效率和 Attention 的全局标定
# =====================================================
class HybridBlock(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        # 1. 双向 Mamba 支路
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

        # 2. 自注意力支路 (用于锁定瞬时病理特征)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 3. 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Mamba 扫描
        res = x
        x1 = self.ln1(x)
        # 正向 + 反向 (Flip 技巧实现双向)
        x_m = self.mamba_fwd(x1) + torch.flip(self.mamba_bwd(torch.flip(x1, [1])), [1])
        x = res + self.dropout(x_m)

        # Attention 标定
        res = x
        x_a, _ = self.attn(self.ln2(x), self.ln2(x), self.ln2(x))
        x = res + self.dropout(x_a)

        # MLP
        x = x + self.dropout(self.mlp(x))
        return x


# =====================================================
# 第三部分：整机架构 (VimA-Hybrid)
# =====================================================
class VimAHybrid(nn.Module):
    def __init__(self, num_classes=1, n_layers=6, d_model=192, patch_time=4):
        super().__init__()
        # 1. 卷积前端
        self.stem = AcousticStripStem(freq_bins=128, patch_time=patch_time, embed_dim=d_model)

        # 2. 可学习的位置编码 (256 = 1024 / 4)
        num_patches = 1024 // patch_time
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # 3. 堆叠混合层
        self.blocks = nn.ModuleList([
            HybridBlock(d_model=d_model) for _ in range(n_layers)
        ])

        # 4. 输出头
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 1, 128, 1024]
        x = self.stem(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.head(x).squeeze(-1)  # 输出用于 BCE 损失