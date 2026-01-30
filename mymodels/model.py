import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================
# 1) VisionMambaBlock: 模仿 MambaVision 的改进块
# =====================================================
class VisionMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        # 核心改进：Mamba 内部集成 dw_conv (通过 Mamba 原生 d_conv 参数)
        # 加上我们自定义的并行 DW-Conv 路径增强局部感知
        self.fwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

        # 深度可分离卷积分支：专门抓取声谱图细微纹理
        self.dw_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # --- 序列处理分支 ---
        res = x
        x_norm = self.ln1(x)

        # 1. 双向 Mamba 路径
        mamba_feat = self.fwd_mamba(x_norm) + torch.flip(self.bwd_mamba(torch.flip(x_norm, [1])), [1])

        # 2. 局部卷积路径 (DW-Conv)
        local_feat = self.dw_conv(x_norm.transpose(1, 2)).transpose(1, 2)

        # 融合残差
        x = res + mamba_feat + local_feat

        # --- 前馈网络分支 ---
        x = x + self.mlp(self.ln2(x))
        return x


# =====================================================
# 2) SSA_Layer: 并行混合层 (MambaVision 核心思想)
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        # 分支 A: Mamba 序列建模堆栈 (3层)
        self.mamba_stack = nn.Sequential(*[
            VisionMambaBlock(d_model, dropout=dropout) for _ in range(3)
        ])

        # 分支 B: Multi-Head Attention 全局检索
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 分支 C: 额外的 Post-Mamba Refining (你要求的 3 层追加)
        self.post_mamba = nn.Sequential(*[
            VisionMambaBlock(d_model, dropout=dropout) for _ in range(3)
        ])

        # 融合门控：动态决定 Mamba 和 Attention 的权重
        self.fusion_gate = nn.Parameter(torch.ones(d_model) * 0.5)

        self.layer_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_layer = x
        x_norm = self.norm(x)

        # 1. 并行处理：Mamba vs Attention
        # Mamba 负责捕捉时间因果关系
        x_m = self.mamba_stack(x_norm)

        # Attention 负责捕捉全局病理证据
        x_a, _ = self.attn(x_norm, x_norm, x_norm)

        # 2. 动态融合
        gate = torch.sigmoid(self.fusion_gate)
        x_combined = x_m * gate + x_a * (1 - gate)

        # 3. 后置处理：再次通过 Mamba 精炼特征
        x_refined = self.post_mamba(x_combined)

        # 4. 门控残差输出
        g = self.layer_gate(x_refined.mean(dim=1, keepdim=True))
        return res_layer + g * x_refined


# =====================================================
# 3) SSA_Model: 顶层模型架构 (Top-K 证据检索)
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

        # 池化权重生成
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # 初始投影
        x = self.input_proj(x) + self.pos_embed
        x = self.pos_drop(x)

        # 逐层演进
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # --- Top-K Evidence Attention (基于你的设想改进) ---
        attn_logits = self.pool_proj(x)  # [B, L, 1]

        # 只选取注意力最高的前 16 个“证据 Patch”
        _, topk_idx = torch.topk(attn_logits.squeeze(-1), k=16, dim=1)
        mask = torch.full_like(attn_logits, -1e9)
        mask.scatter_(1, topk_idx.unsqueeze(-1), 0)

        # 证据聚合权重 (温度系数 0.5 让模型更果断)
        topk_weights = torch.softmax((attn_logits + mask) / 0.5, dim=1)

        # 三路融合：聚合特征、局部最大、全局平均
        attn_feat = torch.sum(x * topk_weights, dim=1)
        max_feat, _ = torch.max(x, dim=1)
        avg_feat = torch.mean(x, dim=1)

        feat = torch.cat([attn_feat, max_feat, avg_feat], dim=-1)

        return self.head(feat).squeeze(-1)


# =====================================================
# 构建函数
# =====================================================
def build_model(input_dim=1024, d_model=256, dropout=0.15, num_classes=1, seq_len=96):
    return SSA_Model(
        input_dim=input_dim,
        d_model=d_model,
        dropout=dropout,
        num_classes=num_classes,
        seq_len=seq_len,
    )