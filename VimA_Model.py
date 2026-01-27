import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm causal-conv1d)")
    exit()


# =====================================================
# 1) 改进的 Stem：使用 3x3 小卷积堆叠 (ResNet-style)
# 相比单层条带卷积，这能更好地捕获频谱图的局部特征
# =====================================================
class ConvStem(nn.Module):
    def __init__(self, embed_dim=192, patch_time=4):
        super().__init__()
        # 第一层：保持尺寸或轻微下采样
        # [B, 1, 128, 1024] -> [B, 64, 64, 512]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # 第二层：关键修复
        # 我们要让高度从 64 变为 1
        # 宽度（时间轴）的步长由 patch_time 决定
        # 既然第一层已经 stride=2 了，这里只需要再 stride = patch_time // 2
        stride_w = max(1, patch_time // 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=(64, 3), stride=(64, stride_w), padding=(0, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv1(x)  # 输出 H=64, W=512
        x = self.conv2(x)  # 输出 H=1, W=512/(patch_time//2)

        # 整理形状为 [B, L, D]
        x = x.squeeze(2)  # [B, D, L]
        x = x.transpose(1, 2)  # [B, L, D]
        return self.norm(x)


# =====================================================
# 2) 混合层 HybridBlock (保持逻辑，优化 Dropout)
# =====================================================
class HybridBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):  # 默认提高到 0.3
        super().__init__()
        # Mamba 支路
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)  # 减小 d_state
        self.mamba_bwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.drop_m = nn.Dropout(dropout)

        # Attention 支路
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop_a = nn.Dropout(dropout)

        # MLP 支路
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Bi-Mamba
        res = x
        x1 = self.ln1(x)
        x_m = self.mamba_fwd(x1) + torch.flip(self.mamba_bwd(torch.flip(x1, [1])), [1])
        x = res + self.drop_m(x_m)

        # Attention
        res = x
        x_a, _ = self.attn(self.ln2(x), self.ln2(x), self.ln2(x))
        x = res + self.drop_a(x_a)

        # MLP
        x = x + self.mlp(self.ln3(x))
        return x


# =====================================================
# 3) 整机架构 VimAHybrid (移除 CLS, 采用 Attention Pooling)
# =====================================================
class VimAHybrid(nn.Module):
    def __init__(self, num_classes=1, n_layers=6, d_model=192, patch_time=4, dropout=0.3):
        super().__init__()
        # 1. 卷积前端
        self.stem = ConvStem(embed_dim=d_model, patch_time=patch_time)

        # 2. 移除 CLS Token, 只保留位置编码
        # 根据卷积后的实际序列长度计算
        num_patches = 1024 // patch_time
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # 3. 堆叠混合层
        self.blocks = nn.ModuleList([
            HybridBlock(d_model=d_model, dropout=dropout) for _ in range(n_layers)
        ])

        # 4. Attention Pooling 层
        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, 1)  # 计算每个 patch 的权重

        # 5. 分类头
        self.head = nn.Linear(d_model, num_classes)

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, 1, 128, 1024]
        x = self.stem(x)  # [B, L, D]
        B, L, D = x.shape
        # 对齐位置编码（防止由于下采样导致的尺寸微差）
        x = x + self.pos_embed[:, :x.size(1), :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)  # [B, L, D]

        # --- Attention Pooling ---
        # 计算每个时间步的重要性分数
        weights = torch.softmax(self.pool_proj(x), dim=1)  # [B, L, 1]
        # 加权平均：把序列维度 L 消除
        feat = torch.sum(x * weights, dim=1)  # [B, D]

        return self.head(feat).squeeze(-1)  # [B]