import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# =====================================================
# 2) SSA_Layer: 3个 BiMamba + 1个 Attention (大层单元)
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # 内部堆叠 3 层 BiMamba
        self.bimamba_stack = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # 内部 1 层全局 Attention
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 连续经过 3 个 BiMamba (时序深度建模)
        for bimamba in self.bimamba_stack:
            x = bimamba(x)

        # 2. 经过 1 个 Attention (全局特征精炼)
        res = x
        x_norm = self.attn_ln(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + self.drop(x_attn)
        return x


# =====================================================
# 3) SSA_Model: 顶层架构 (含一维卷积前端 + 4个 SSA_Layer)
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, d_model=256, dropout=0.2, n_layers=4):
        super().__init__()

        # A. 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)

        # B. 【一维卷积前端】 在 Mamba 扫描前捕获局部局部声学特征
        self.conv_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),  # Depthwise
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),  # Pointwise
            nn.Dropout(dropout)
        )

        # C. 核心架构堆叠：4 个 SSA_Layer (共计 12 BiMamba + 4 Attention)
        self.layers = nn.ModuleList([
            SSA_Layer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        # D. 归一化与混合池化分类头
        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # 1. 投影与 1D 卷积处理 (注意维度转换 [B, L, C] -> [B, C, L])
        x = self.input_proj(x)
        x_conv = x.transpose(1, 2)
        x = self.conv_stem(x_conv).transpose(1, 2)

        # 2. 核心堆叠层计算
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # 3. 混合池化 (Attention Pooling + Global Max Pooling)
        weights = torch.softmax(self.pool_proj(x), dim=1)
        attn_feat = torch.sum(x * weights, dim=1)
        max_feat, _ = torch.max(x, dim=1)

        feat = torch.cat([attn_feat, max_feat], dim=-1)  # [B, d_model * 2]

        return self.head(feat).squeeze(-1)