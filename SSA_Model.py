import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================
# 1) HybridLayer: 3层 Mamba + 1层 Attention 的组合单元
# =====================================================
class HybridLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # 内部 3 层 Mamba
        self.mamba_sublayers = nn.ModuleList([
            MambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # 内部 1 层 Attention
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 连续经过 3 个 Mamba 块
        for mamba_blk in self.mamba_sublayers:
            x = mamba_blk(x)

        # 2. 经过 1 个 Attention 块
        res = x
        x_norm = self.attn_ln(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + self.drop(x_attn)
        return x


# =====================================================
# 2) SSA_Model: 堆叠 4 个 HybridLayer
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, d_model=256, dropout=0.2, n_hybrid_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        # 卷积 Stem (保持在最前端，处理局部特征)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        # 核心层：堆叠 4 个 HybridLayer
        # 总深度：12 Mamba + 4 Attention
        self.hybrid_stack = nn.ModuleList([
            HybridLayer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_hybrid_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # 1. 预处理
        x = self.input_proj(x)
        x_conv = x.transpose(1, 2)
        x = self.conv_stem(x_conv).transpose(1, 2)

        # 2. 堆叠循环：依次通过 4 个混合层
        for layer in self.hybrid_stack:
            x = layer(x)

        # 3. 归一化与池化
        x = self.norm(x)
        weights = torch.softmax(self.pool_proj(x), dim=1)
        attn_feat = torch.sum(x * weights, dim=1)
        max_feat, _ = torch.max(x, dim=1)

        feat = torch.cat([attn_feat, max_feat], dim=-1)
        return self.head(feat).squeeze(-1)