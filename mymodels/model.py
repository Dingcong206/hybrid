import torch
import torch.nn as nn

# =============== RMSNorm 兼容（torch 版本不够会没有 nn.RMSNorm） ===============
def _rmsnorm(dim: int):
    return nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)

# =============== mamba 依赖（没有就提示） ===============
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("⚠️ 未安装 mamba_ssm。需要的话安装：pip install mamba-ssm causal-conv1d")

# =========================
# 1) 位置编码函数（你缺的就是这个）
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
# 2) BiMambaBlock（你也缺了这个）
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
        # x: (B,T,D)
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        return x + self.mlp(self.ln2(x))

# =========================
# 3) SE Block（你写的 그대로）
# =========================
class SE_Block(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, D)
        b, t, d = x.size()
        y = torch.mean(x, dim=1)            # (B,D)
        y = self.fc(y).view(b, 1, d)        # (B,1,D)
        return x * y.expand_as(x)

# =========================
# 4) 改进后的 SSA Layer（你写的 + 加一个 mask-aware mean 可选）
# =========================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

        self.mamba_pre = BiMambaBlock(d_model, dropout=dropout)

        self.attn_ln = _rmsnorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.se = SE_Block(d_model)

    def forward(self, x, mask=None):
        # 1) local conv
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # 2) bimamba
        x = self.mamba_pre(x)

        # 3) attention
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        x = x + x_a

        # 4) SE
        x = self.se(x)
        return x

# =========================
# 5) ICBHI 专用 Pooling（你写的 그대로）
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
        # x: (B,T,D)
        attn_scores = self.attn_net(x)  # (B,T,1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)  # (B,D)

        if mask is not None:
            x_for_max = x.masked_fill(mask.unsqueeze(-1), -1e9)
        else:
            x_for_max = x
        feat_max, _ = torch.max(x_for_max, dim=1)     # (B,D)

        return feat_weighted + feat_max

# =========================
# 6) 主模型：支持 return_feature（路线A用）
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(
        self,
        in_dim=768,
        d_model=256,
        n_layers=4,
        nhead=8,
        dropout=0.3,
        max_len=4096,
        num_classes=4,
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

        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = _rmsnorm(d_model)
        self.pool = ICBHI_Pooling(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.token_head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None, return_feature=False):
        # x: (B,T,in_dim)
        x = self.input_proj(x)
        T = x.shape[1]
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)

        file_feature = self.pool(x, mask=mask)  # (B,D)

        if return_feature:
            return file_feature, x

        file_logits = self.classifier(file_feature)
        token_logits = self.token_head(x)
        return file_logits, token_logits

# =========================
# 7) SSA_Backbone（你缺的就是这个）
# =========================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_HeARTokens):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model

    def forward(self, x, mask=None):
        feat, _ = self.ssa(x, mask=mask, return_feature=True)  # (B,D)
        return feat

# =========================
# 8) 工厂函数：build_backbone（路线A训练用）
# =========================
def build_backbone(in_dim=768, d_model=256, n_layers=4, nhead=8, dropout=0.3, max_len=4096):
    ssa = SSA_Model_HeARTokens(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
        num_classes=4,
    )
    backbone = SSA_Backbone(ssa)

    params = sum(p.numel() for p in backbone.parameters())
    print(f"✅ ICBHI Optimized SSA Backbone Initialized.")
    print(f"   Parameters: {params:,} | Feature Dim: {backbone.final_feat_dim}")
    return backbone
