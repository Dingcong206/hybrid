import torch
import torch.nn as nn

# =============== RMSNorm 兼容（torch 版本不够会没有 nn.RMSNorm） ===============
def _rmsnorm(dim: int):
    return nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)

# =============== mamba 依赖 ===============
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("⚠️ 未安装 mamba_ssm。需要安装：pip install mamba-ssm causal-conv1d")

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
# 2) BiMambaBlock（✅ 无 MLP，仅 SSM 残差）
# =========================
class BiMambaBlock(nn.Module):
    """
    纯 BiMamba：LN -> (fwd+bwd) -> Dropout -> residual
    ✅ 不包含任何 MLP/FFN（你要求的）
    """
    def __init__(self, d_model, dropout=0.2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise RuntimeError(
                "当前环境没有安装 mamba_ssm，无法使用 BiMambaBlock。\n"
                "请安装：pip install mamba-ssm causal-conv1d"
            )

        self.ln = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,T,D)
        h = self.ln(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        return x + self.drop(h_f + h_b)

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
        if mask is not None:
            x_n = x_n.masked_fill(mask.unsqueeze(-1), 0.0)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        return x + self.drop(x_a)

# =========================
# 4) FFNBlock（残差）——只在 stage 末尾用
# =========================
class FFNBlock(nn.Module):
    def __init__(self, d_model, dropout=0.3, mult=4):
        super().__init__()
        self.ln = _rmsnorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.ln(x))

# =========================
# 5) ICBHI Pooling：Attention pooling + Max pooling
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
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)  # (B,D)

        if mask is not None:
            x_for_max = x.masked_fill(mask.unsqueeze(-1), -1e9)
        else:
            x_for_max = x
        feat_max, _ = torch.max(x_for_max, dim=1)  # (B,D)

        return feat_weighted + feat_max

# =========================
# 6) 一个 Stage（= 可堆叠层）
#    Attn -> 6×BiMamba(no-MLP) -> Attn -> FFN
#    ✅ 注意：这里没有 Conv（Conv 只在最开始）
# =========================
class StageAttn6BiMambaAttnFFN(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        dropout=0.3,
        n_bimamba=6,
        d_state=16,
        d_conv=4,
        expand=2,
        ffn_mult=4,
    ):
        super().__init__()
        self.attn1 = AttentionBlock(d_model, nhead=nhead, dropout=dropout)

        self.bimambas = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_bimamba)
        ])

        self.attn2 = AttentionBlock(d_model, nhead=nhead, dropout=dropout)
        self.ffn = FFNBlock(d_model, dropout=dropout, mult=ffn_mult)

    def forward(self, x, mask=None):
        x = self.attn1(x, mask=mask)
        for blk in self.bimambas:
            x = blk(x)
        x = self.attn2(x, mask=mask)
        x = self.ffn(x)
        return x

# =========================
# 7) 主模型：PE -> Conv(一次) -> [Stage × n_layers] -> Norm -> Pool
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(
        self,
        in_dim=768,
        d_model=256,
        n_layers=2,
        nhead=8,
        dropout=0.3,
        max_len=4096,
        num_classes=4,
        conv_k=7,
        n_bimamba=6,     # 每层 6 个 BiMamba
        d_state=16,
        d_conv=4,
        expand=2,
        ffn_mult=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            _rmsnorm(d_model),
        )

        pe = sinusoidal_positional_encoding(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        # ✅ Conv 只在最开始一次
        self.front_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_k, padding=conv_k // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

        # ✅ 堆叠 stages
        self.stages = nn.ModuleList([
            StageAttn6BiMambaAttnFFN(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                n_bimamba=n_bimamba,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ffn_mult=ffn_mult,
            )
            for _ in range(n_layers)
        ])

        self.norm = _rmsnorm(d_model)
        self.pool = ICBHI_Pooling(d_model)

        # 兼容路线B（Route-A 不用，但保留无害）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.token_head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None, return_feature=False):
        x = self.input_proj(x)  # (B,T,D)
        T = x.shape[1]
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        # ✅ Conv only once
        x = x + self.front_conv(x.transpose(1, 2)).transpose(1, 2)

        for stage in self.stages:
            x = stage(x, mask=mask)

        x = self.norm(x)
        file_feature = self.pool(x, mask=mask)  # (B,D)

        if return_feature:
            return file_feature, x

        file_logits = self.classifier(file_feature)
        token_logits = self.token_head(x)
        return file_logits, token_logits

# =========================
# 8) Route-A Backbone：输出 (B,d_model)
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
# 9) build_backbone（Route-A）
# =========================
def build_backbone(
    in_dim=768,
    d_model=256,
    n_layers=2,
    nhead=8,
    dropout=0.3,
    max_len=4096,
    conv_k=7,
    d_state=16,
    d_conv=4,
    expand=2,
    ffn_mult=4,
):
    ssa = SSA_Model_HeARTokens(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
        num_classes=4,
        conv_k=conv_k,
        n_bimamba=6,  # ✅ 固定每层 6 个
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        ffn_mult=ffn_mult,
    )
    backbone = SSA_Backbone(ssa)

    params = sum(p.numel() for p in backbone.parameters())
    print(f"✅ Structure: PE → Conv(once) → [Attn → 6×BiMamba(no-MLP) → Attn → FFN] × {n_layers}")
    print(f"   Parameters: {params:,} | Feature Dim: {backbone.final_feat_dim}")
    return backbone
