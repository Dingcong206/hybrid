import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =====================================================
# 1) BiMambaBlock: 双向 Mamba 核心块
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
        # Mamba 残差分支
        h = self.ln1(x)
        # 双向融合：前向 + 翻转后的后向再翻转回来
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        # FFN 残差分支
        x = x + self.mlp(self.ln2(x))
        return x


# =====================================================
# 2) SSA_Layer: 空间-频谱-注意力层 (核心修改处)
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=12, dropout=0.3):
        super().__init__()

        # A. 卷积层：提取局部纹理
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        # B. Pre-Attention Mamba 堆栈 (3层)
        self.bimamba_stack = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # C. Attention 层
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)

        # D. 新增：Post-Attention Mamba 堆栈
        self.post_mamba_stack = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # E. 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_layer = x  # 大残差

        # 1. 卷积：处理局部信息
        x_conv = x.transpose(1, 2)
        x = x + self.conv(x_conv).transpose(1, 2)

        # 2. Pre-Attention Mamba：时序初步建模
        for bimamba in self.bimamba_stack:
            x = bimamba(x)

        # 3. Attention：全局特征检索
        res_attn = x
        x_norm = self.attn_ln(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x_out = res_attn + self.attn_drop(x_attn)

        # 4. Post-Attention Mamba：追加的 3 层 Mamba 进行精炼
        for post_bimamba in self.post_mamba_stack:
            x_out = post_bimamba(x_out)

        # 5. 应用门控
        g = self.gate(x_out.mean(dim=1, keepdim=True))
        return res_layer + g * x_out


# =====================================================
# 3) SSA_Model: 顶层架构
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, d_model=256, dropout=0.2, n_layers=2, seq_len=96):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.pos_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SSA_Layer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

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
        x = self.input_proj(x) + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Top-K 选通注意力池化
        attn_logits = self.pool_proj(x)
        _, topk_idx = torch.topk(attn_logits.squeeze(-1), k=16, dim=1)
        mask = torch.zeros_like(attn_logits).fill_(-1e9)
        mask.scatter_(1, topk_idx.unsqueeze(-1), 0)
        topk_weights = torch.softmax((attn_logits + mask) / 0.5, dim=1)

        # 三路融合
        attn_feat = torch.sum(x * topk_weights, dim=1)
        max_feat, _ = torch.max(x, dim=1)
        avg_feat = torch.mean(x, dim=1)

        feat = torch.cat([attn_feat, max_feat, avg_feat], dim=-1)

        return self.head(feat).squeeze(-1)


def build_model(input_dim=1024, d_model=256, dropout=0.15, num_classes=1, seq_len=96):
    return SSA_Model(
        input_dim=input_dim,
        d_model=d_model,
        dropout=dropout,
        num_classes=num_classes,
        seq_len=seq_len,
    )