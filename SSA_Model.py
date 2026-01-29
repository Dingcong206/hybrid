import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================
# 1) BiMambaBlock: 双向扫描核心单元 (基础零件)
# =====================================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # 前向与后向 Mamba
        self.fwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x_norm = self.ln(x)
        # 并行计算双向特征
        f_out = self.fwd_mamba(x_norm)
        b_out = torch.flip(self.bwd_mamba(torch.flip(x_norm, [1])), [1])
        return res + self.drop(f_out + b_out)


# =====================================================
# 2) SSA_Layer: 1个 1D卷积 + 3个 BiMamba + 1个 Attention
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # A. 局部特征聚合 (1D Convolution)
        # 放在 Mamba 之前，每一层都先重新梳理局部邻域特征
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        # B. 内部堆叠 3 层 BiMamba (时序特征挖掘)
        self.bimamba_stack = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # C. 内部 1 层全局 Attention (全局关联精炼)
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 先进行一维卷积处理 (维度转换 [B, L, C] -> [B, C, L])
        res_conv = x
        x_conv = x.transpose(1, 2)
        x = self.conv(x_conv).transpose(1, 2)
        x = x + res_conv  # 残差连接

        # 2. 连续经过 3 个 BiMamba
        for bimamba in self.bimamba_stack:
            x = bimamba(x)

        # 3. 经过 1 个 Attention
        res_attn = x
        x_norm = self.attn_ln(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = res_attn + self.drop(x_attn)

        return x


# =====================================================
# 3) SSA_Model: 顶层架构 (堆叠 4 个 SSA_Layer)
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, d_model=256, dropout=0.2, n_layers=4):
        super().__init__()

        # A. 输入投影层：将 HeAR 的 1024 维映射到 d_model 维度
        self.input_proj = nn.Linear(input_dim, d_model)

        # B. 核心架构堆叠：4 个 SSA_Layer
        # 每个大层内部结构：[Conv -> 3xBiMamba -> Attention]
        self.layers = nn.ModuleList([
            SSA_Layer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        # C. 归一化与分类头
        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # 1. 投影层
        x = self.input_proj(x)

        # 2. 依次通过 4 个 SSA_Layer 大层
        for layer in self.layers:
            x = layer(x)

        # 3. 最终归一化
        x = self.norm(x)

        # 4. 混合池化 (Attention Pooling + Global Max Pooling)
        # 这种方式对呼吸音中的突发杂音（爆裂音）非常有效
        weights = torch.softmax(self.pool_proj(x), dim=1)
        attn_feat = torch.sum(x * weights, dim=1)
        max_feat, _ = torch.max(x, dim=1)

        feat = torch.cat([attn_feat, max_feat], dim=-1)  # [B, d_model * 2]

        return self.head(feat).squeeze(-1)