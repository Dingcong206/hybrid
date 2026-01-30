import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================
# 1) BiMambaBlock: 保持双向逻辑
# =====================================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Mamba 残差
        h = self.ln1(x)
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        # FFN 残差
        x = x + self.mlp(self.ln2(x))
        return x


# =====================================================
# 2) SSA_Layer: 增强卷积表达与门控连接
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # A. 改进卷积：增大感受野，不使用 groups 以增强通道间交互
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        self.bimamba_stack = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

        # 门控：动态调节当前层特征贡献
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_conv = x
        x_conv = x.transpose(1, 2)
        x_c = self.conv(x_conv).transpose(1, 2)
        x = res_conv + x_c

        for bimamba in self.bimamba_stack:
            x = bimamba(x)

        res_attn = x
        x_norm = self.attn_ln(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x_out = res_attn + self.drop(x_attn)

        # 应用门控
        g = self.gate(x_out.mean(dim=1, keepdim=True))
        return res_conv + g * x_out


# =====================================================
# 3) SSA_Model: 顶层架构 (引入 Top-K 池化与三路融合)
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, d_model=256, dropout=0.2, n_layers=4, seq_len=96):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.pos_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SSA_Layer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 非线性池化投射
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        # 分类头：支持三路特征融合 (d_model * 3)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # --- 核心改进：Top-K 选通注意力池化 ---
        attn_logits = self.pool_proj(x)  # [B, L, 1]

        # 增加温度系数 0.5 使权重更集中
        weights = torch.softmax(attn_logits / 0.5, dim=1)

        # 只取注意力最强的 16 个 Patch (Top-K)
        _, topk_idx = torch.topk(attn_logits.squeeze(-1), k=16, dim=1)
        # 掩码逻辑：将非 Top-K 的权重设为极小
        mask = torch.zeros_like(attn_logits).fill_(-1e9)
        mask.scatter_(1, topk_idx.unsqueeze(-1), 0)
        topk_weights = torch.softmax((attn_logits + mask) / 0.5, dim=1)

        # 三路特征提取
        attn_feat = torch.sum(x * topk_weights, dim=1)  # Top-K 注意力聚合
        max_feat, _ = torch.max(x, dim=1)  # 全局最大特征
        avg_feat = torch.mean(x, dim=1)  # 全局平均背景

        # 融合特征
        feat = torch.cat([attn_feat, max_feat, avg_feat], dim=-1)  # [B, d_model * 3]

        return self.head(feat).squeeze(-1)