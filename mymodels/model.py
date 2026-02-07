import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

# =========================
# 0) 基础组件：RMSNorm 兼容性
# =========================
def _rmsnorm(dim: int):
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    else:
        return CustomRMSNorm(dim)

class CustomRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

# =========================
# 1) 位置编码
# =========================
def sinusoidal_positional_encoding(seq_len: int, dim: int, device="cpu"):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# =========================
# 2) BiMambaBlock (1层双向扫描)
# =========================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("请安装 mamba-ssm")

        self.ln1 = _rmsnorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

        self.ln2 = _rmsnorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 时序扫描分支
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        # 局部表征分支
        x = x + self.mlp(self.ln2(x))
        return x

# =========================
# 3) AttentionBlock
# =========================
class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        self.ln = _rmsnorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_n = self.ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        return x + self.drop(x_a)

# =========================
# 4) ICBHI Pooling (Attn + Mean)
# =========================
class ICBHI_Pooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, mask=None):
        attn_scores = self.attn_net(x)
        if mask is not None:
            neg_inf = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), neg_inf)

        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)

        if mask is not None:
            x_valid = x.masked_fill(mask.unsqueeze(-1), 0.0)
            denom = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            feat_mean = x_valid.sum(dim=1) / denom
        else:
            feat_mean = x.mean(dim=1)

        return feat_weighted + feat_mean

# =========================
# 5) Stage：1 Mamba + 1 Attention
# =========================
class Stage1M1A(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
        self.attn = AttentionBlock(d_model, nhead=nhead, dropout=dropout)

    def forward(self, x, mask=None):
        x = self.mamba(x)
        x = self.attn(x, mask=mask)
        return x

# =========================
# 6) 主模型 SSA-Net (1:1 交替版)
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(self, in_dim=768, d_model=512, n_layers=8, nhead=8, dropout=0.3, max_len=1024, num_classes=1):
        super().__init__()
        # 输入映射
        self.input_proj = nn.Sequential(
            _rmsnorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.SiLU()
        )

        # 位置编码
        pe = sinusoidal_positional_encoding(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        # 初始卷积增强
        self.front_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=1, bias=False),
            nn.SiLU(),
        )
        self.front_ln = _rmsnorm(d_model)

        # 核心 Stages: 1 Mamba + 1 Attention 交替
        self.stages = nn.ModuleList([
            Stage1M1A(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = _rmsnorm(d_model)
        self.pool = ICBHI_Pooling(d_model)

        # 分类头 (二分类默认输出 1 维进行 sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None, return_feature=False):
        # B x T x In_Dim -> B x T x D_model
        x = self.input_proj(x)
        T = x.shape[1]
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        # 卷积提取局部纹理
        y = self.front_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.front_ln(y)

        # 8层交替迭代
        for stage in self.stages:
            x = stage(x, mask=mask)

        x = self.final_norm(x)
        file_feature = self.pool(x, mask=mask)

        if return_feature:
            return file_feature, x

        return self.classifier(file_feature)

# =========================
# 7) Backbone 包装类
# =========================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_HeARTokens):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model

    def forward(self, x, mask=None):
        feat, _ = self.ssa(x, mask=mask, return_feature=True)
        return feat

def build_model(in_dim=768, d_model=512, n_layers=4, nhead=8, num_classes=1):
    ssa = SSA_Model_HeARTokens(in_dim=in_dim, d_model=d_model, n_layers=n_layers, nhead=nhead, num_classes=num_classes)
    backbone = SSA_Backbone(ssa)
    params = sum(p.numel() for p in backbone.parameters())
    print(f"Structure: RMSNorm + 1:1 Mamba-Attn Stage x {n_layers}")
    print(f"Total Params: {params:,} | Feature Dim: {backbone.final_feat_dim}")
    return backbone
# ============================================================
# ✅ 模仿旧版：添加别名，确保兼容 __init__.py 的导入
# ============================================================
SSA_Model = SSA_Model_HeARTokens  # 让旧代码能找到 SSA_Model
build_backbone = build_model      # 如果旧版用到了这个名字