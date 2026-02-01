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
        h = self.ln1(x)
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        x = x + self.mlp(self.ln2(x))
        return x


# =====================================================
# 2) SSA_Layer: 卷积 + (Pre Mamba) + Attention + (Post Mamba) + Gate
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # A. 卷积层：提取局部纹理
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        # B. Pre-Attention Mamba 堆栈
        self.bimamba_stack = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # C. Attention 层
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)

        # D. Post-Attention Mamba 堆栈
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
        res_layer = x

        # 1) Conv
        x_conv = x.transpose(1, 2)
        x = x + self.conv(x_conv).transpose(1, 2)

        # 2) Pre Mamba
        for bimamba in self.bimamba_stack:
            x = bimamba(x)

        # 3) Attention
        res_attn = x
        x_norm = self.attn_ln(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x_out = res_attn + self.attn_drop(x_attn)

        # 4) Post Mamba
        for post_bimamba in self.post_mamba_stack:
            x_out = post_bimamba(x_out)

        # 5) Gate
        g = self.gate(x_out.mean(dim=1, keepdim=True))
        return res_layer + g * x_out


# =====================================================
# 3) SSA_Model: 顶层架构（加入显式升维 + 可变长位置编码）
# =====================================================
class SSA_Model(nn.Module):
    def __init__(
        self,
        input_dim=48,
        num_classes=1,
        d_model=256,
        dropout=0.2,
        n_layers=2,
        nhead=8,
        max_seq_len=200,   # 用于初始化位置编码长度（你现在 L=200）
        topk=16
    ):
        super().__init__()

        assert d_model % nhead == 0, f"d_model={d_model} 必须能被 nhead={nhead} 整除"

        # ✅ 0) 显式 Linear 升维（48 -> d_model）
        self.linear_up = nn.Linear(input_dim, d_model)

        # ✅ 1) 可学习位置编码（长度先按 max_seq_len 初始化）
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.pos_drop = nn.Dropout(dropout)

        # 2) 编码层堆叠
        self.layers = nn.ModuleList([
            SSA_Layer(d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 3) Top-K pooling
        self.topk = topk
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        # 4) 分类头（attn/max/avg 三路融合）
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _add_pos_embed(self, x):
        """
        x: (B, L, d_model)
        pos_embed: (1, max_seq_len, d_model)
        如果 L != max_seq_len，就插值到 L
        """
        B, L, D = x.shape
        pos = self.pos_embed

        if pos.shape[1] != L:
            # 插值需要 (B, D, L)
            pos = pos.transpose(1, 2)  # (1, D, maxL)
            pos = F.interpolate(pos, size=L, mode="linear", align_corners=False)
            pos = pos.transpose(1, 2)  # (1, L, D)

        return x + pos

    def forward(self, x):
        """
        输入 x: (B, L, 48) 例如 (B, 200, 48)
        """
        # ✅ 先升维到 d_model
        x = self.linear_up(x)  # (B, L, d_model)

        # ✅ 加位置编码（支持可变 L）
        x = self._add_pos_embed(x)
        x = self.pos_drop(x)

        # 编码层
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Top-K 选通注意力池化
        attn_logits = self.pool_proj(x)  # (B, L, 1)
        k = min(self.topk, x.shape[1])
        _, topk_idx = torch.topk(attn_logits.squeeze(-1), k=k, dim=1)

        mask = torch.zeros_like(attn_logits).fill_(-1e9)
        mask.scatter_(1, topk_idx.unsqueeze(-1), 0)
        topk_weights = torch.softmax((attn_logits + mask) / 0.5, dim=1)  # (B, L, 1)

        # 三路融合
        attn_feat = torch.sum(x * topk_weights, dim=1)
        max_feat, _ = torch.max(x, dim=1)
        avg_feat = torch.mean(x, dim=1)

        feat = torch.cat([attn_feat, max_feat, avg_feat], dim=-1)
        return self.head(feat).squeeze(-1)


def build_model(
    input_dim=48,
    d_model=256,
    dropout=0.2,
    num_classes=1,
    n_layers=2,
    nhead=8,
    max_seq_len=200,
    topk=16
):
    return SSA_Model(
        input_dim=input_dim,
        d_model=d_model,
        dropout=dropout,
        num_classes=num_classes,
        n_layers=n_layers,
        nhead=nhead,
        max_seq_len=max_seq_len,
        topk=topk
    )
