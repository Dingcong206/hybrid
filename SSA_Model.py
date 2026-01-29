import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
# =====================================================
# 1) MambaBlock: 纯粹的 Mamba 时序建模块
# =====================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # 双向 Mamba 捕捉呼吸音的上下文
        self.fwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x_norm = self.ln(x)
        f_out = self.fwd_mamba(x_norm)
        # 反向扫描
        b_out = torch.flip(self.bwd_mamba(torch.flip(x_norm, [1])), [1])
        return res + self.drop(f_out + b_out)


# =====================================================
# 2) SSA_Model: 3+1 混合架构
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, d_model=256, dropout=0.2):
        super().__init__()

        # A. 输入投影 + 卷积 Stem
        self.input_proj = nn.Linear(input_dim, d_model)

        # 创新：Mamba 前的卷积，用于聚合 Patch 间的局部声学纹理
        self.conv_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        # B. 核心层：3层双向 Mamba
        # 针对 6898 个样本，3层是平衡表达力与泛化能力的黄金层数
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # C. 顶层：1层全局 Self-Attention
        # 在 Mamba 梳理完序列后，进行最后的一次全局关联分析
        self.final_attn_ln = nn.LayerNorm(d_model)
        self.final_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True, dropout=dropout)

        # D. 分类头
        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # 1. 投影与卷积预处理
        x = self.input_proj(x)
        x_conv = x.transpose(1, 2)
        x = self.conv_stem(x_conv).transpose(1, 2)

        # 2. 经过 3 层 Mamba 建模
        for mamba_blk in self.mamba_layers:
            x = mamba_blk(x)

        # 3. 经过 1 层全局 Attention 提炼
        res = x
        x_attn, _ = self.final_attn(self.final_attn_ln(x), self.final_attn_ln(x), self.final_attn_ln(x))
        x = res + x_attn

        x = self.norm(x)

        # 4. 混合池化 (Attention Pooling + Max Pooling)
        # 这种组合能同时兼顾“全局平均状态”和“瞬时病理冲击”
        weights = torch.softmax(self.pool_proj(x), dim=1)
        attn_feat = torch.sum(x * weights, dim=1)
        max_feat, _ = torch.max(x, dim=1)

        feat = torch.cat([attn_feat, max_feat], dim=-1)  # [B, d_model * 2]

        return self.head(feat).squeeze(-1)