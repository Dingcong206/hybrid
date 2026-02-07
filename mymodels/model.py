import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None
# =============== RMSNorm 兼容 ===============
def _rmsnorm(dim: int):
    return nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)


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
# 2) BiMambaBlock（双向 + FFN）
#   ✅ 保留内部 MLP（更强表征）
# =========================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise RuntimeError(
                "当前环境没有安装 mamba_ssm，无法使用 BiMambaBlock。\n"
                "请安装：pip install mamba-ssm causal-conv1d"
            )

        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        return x + self.mlp(self.ln2(x))


# =========================
# 3) AttentionBlock（残差）
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
# 4) ICBHI Pooling：Attn pooling + Mean pooling
#   ✅ 把 max pooling 改成 mean（更稳、更泛化）
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
        attn_scores = self.attn_net(x)  # (B,T,1)
        if mask is not None:
            neg_inf = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), neg_inf)

        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)  # (B,D)

        # ✅ mean pooling with mask
        if mask is not None:
            x_valid = x.masked_fill(mask.unsqueeze(-1), 0.0)
            denom = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B,1)
            feat_mean = x_valid.sum(dim=1) / denom
        else:
            feat_mean = x.mean(dim=1)

        return feat_weighted + feat_mean


# =========================
# 5) 一个 Stage（= 1 layer）
#    3×BiMamba -> 1×Attn -> 3×BiMamba
#   ✅ 删除 Stage 尾部 FFN（避免 “MLP 过量”）
# =========================
class Stage3Attn3(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        dropout=0.3,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.pre_mambas = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(3)
        ])

        self.attn = AttentionBlock(d_model, nhead=nhead, dropout=dropout)

        self.post_mambas = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(3)
        ])

    def forward(self, x, mask=None):
        for blk in self.pre_mambas:
            x = blk(x)
        x = self.attn(x, mask=mask)
        for blk in self.post_mambas:
            x = blk(x)
        return x


# =========================
# 6) 主模型：PE -> Conv(once) -> [Stage × n_layers] -> Norm -> Pool
#   ✅ 你要的：d_model=512, n_layers=8
#   ✅ Conv 改成 DW + PW（更强表达）
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(
        self,
        in_dim=768,
        d_model=512,     #  512
        n_layers=8,      #  8 layers
        nhead=8,         # 512/8=64 合理
        dropout=0.3,
        max_len=1024,
        num_classes=4,
        conv_k=5,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.SiLU()
        )

        # PE
        pe = sinusoidal_positional_encoding(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        self.front_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_k, padding=conv_k // 2, groups=1, bias=False),
            nn.SiLU(),
        )
        self.front_ln = nn.LayerNorm(d_model)

        # Stages
        self.stages = nn.ModuleList([
            Stage3Attn3(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.norm = _rmsnorm(d_model)
        self.pool = ICBHI_Pooling(d_model)

        # 分类头（保持你原来的“瓶颈层 + dropout”）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None, return_feature=False):
        # x: (B,T,in_dim)
        x = self.input_proj(x)

        T = x.shape[1]
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)
        y = self.front_conv(x.transpose(1, 2)).transpose(1, 2)  # (B,T,C)
        x = x + self.front_ln(y)

        for stage in self.stages:
            x = stage(x, mask=mask)

        x = self.norm(x)
        file_feature = self.pool(x, mask=mask)

        if return_feature:
            return file_feature, x

        logits = self.classifier(file_feature)
        return logits


# =========================
# 7) Backbone：输出 (B, d_model)
# =========================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_HeARTokens):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model

    def forward(self, x, mask=None):
        feat, _ = self.ssa(x, mask=mask, return_feature=True)
        return feat


# =========================
# 8) build_model：返回 backbone（Route-A）
#   ✅ 默认就是 512 / 8layer
# =========================
def build_model(
    in_dim=768,
    d_model=512,     # ✅ 默认 512
    n_layers=8,      # ✅ 默认 8
    nhead=8,         # ✅ 默认 8
    dropout=0.3,
    max_len=1024,
    conv_k=5,
    d_state=16,
    d_conv=4,
    expand=2,
    num_classes=4,
):
    ssa = SSA_Model_HeARTokens(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes,
        conv_k=conv_k,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    backbone = SSA_Backbone(ssa)

    params = sum(p.numel() for p in backbone.parameters())
    print(f"Structure: PE → Conv1D(once, k={conv_k}, groups=1) × {n_layers}")
    print(f"   Parameters: {params:,} | Feature Dim: {backbone.final_feat_dim}")
    return backbone


# ============================================================
# 兼容 mymodels/__init__.py
# ============================================================
SSA_Model = SSA_Model_HeARTokens
build_backbone = build_model
