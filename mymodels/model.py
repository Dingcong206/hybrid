import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# ============================================================
# 0) RMSNorm 兼容
# ============================================================
def _rmsnorm(dim: int):
    return nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else CustomRMSNorm(dim)


class CustomRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


# ============================================================
# 1) 位置编码
# ============================================================
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


# ============================================================
# 2) BiMambaBlock
# ============================================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("请安装 mamba-ssm: pip install mamba-ssm causal-conv1d")

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
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
# 3) AttentionBlock（带 FFN，完整 Transformer Encoder 子块）
# ============================================================
class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=6, dropout=0.2):
        super().__init__()
        self.ln1 = _rmsnorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = _rmsnorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # mask: (B,T) True means padding
        x_n = self.ln1(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        x = x + self.drop1(x_a)
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# 4) Pooling（Attn + Mean）
# ============================================================
class ICBHI_Pooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (B,T,D)
        attn_scores = self.attn_net(x)  # (B,T,1)
        if mask is not None:
            neg_inf = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), neg_inf)

        attn_w = torch.softmax(attn_scores, dim=1)      # (B,T,1)
        feat_weighted = torch.sum(attn_w * x, dim=1)    # (B,D)

        if mask is not None:
            x_valid = x.masked_fill(mask.unsqueeze(-1), 0.0)
            denom = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            feat_mean = x_valid.sum(dim=1) / denom
        else:
            feat_mean = x.mean(dim=1)

        return feat_weighted + feat_mean


# ============================================================
# 5) Layer/Stage：4×BiMamba -> Attn -> 2×BiMamba -> Attn
# ============================================================
class HybridLayer4M1A2M1A(nn.Module):
    def __init__(self, d_model=384, nhead=6, dropout=0.2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.m1 = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(4)
        ])
        self.attn1 = AttentionBlock(d_model, nhead=nhead, dropout=dropout)

        self.m2 = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(2)
        ])
        self.attn2 = AttentionBlock(d_model, nhead=nhead, dropout=dropout)

    def forward(self, x, mask=None):
        for blk in self.m1:
            x = blk(x)
        x = self.attn1(x, mask=mask)
        for blk in self.m2:
            x = blk(x)
        x = self.attn2(x, mask=mask)
        return x


# ============================================================
# 6) 模型本体：输入 tokens (B,T,768) -> 输出 logits (B,num_classes)
#    - D=384, heads=6, N=4 layers
#    - CLS token 可直接包含在 tokens 里（你上游 AST 是否有 CLS 看你生成方式）
# ============================================================
class SSA_HybridTokensModel(nn.Module):
    def __init__(
        self,
        in_dim=768,
        d_model=384,          # ✅ D=384
        n_layers=4,           # ✅ N=4
        nhead=6,              # ✅ heads=6
        dropout=0.2,
        max_len=1024,
        num_classes=4,
        d_state=16,
        d_conv=4,
        expand=2,
        use_front_conv=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            _rmsnorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
        )

        pe = sinusoidal_positional_encoding(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        self.use_front_conv = use_front_conv
        if use_front_conv:
            self.front_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, bias=False),
                nn.SiLU(),
            )
            self.front_ln = _rmsnorm(d_model)

        self.layers = nn.ModuleList([
            HybridLayer4M1A2M1A(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = _rmsnorm(d_model)
        self.pool = ICBHI_Pooling(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, tokens, mask=None, return_feature=False):
        """
        tokens: (B,T,768)  —— 你 AST projection/embedding 后的 tokens
        mask:   (B,T) bool —— True 表示 padding token（可选）
        """
        x = self.input_proj(tokens)  # (B,T,384)

        T = x.shape[1]
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        if self.use_front_conv:
            y = self.front_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.front_ln(y)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        feat = self.pool(x, mask=mask)  # (B,384)

        if return_feature:
            return feat, x

        logits = self.classifier(feat)  # (B,num_classes)
        return logits


def build_model(
    in_dim=768,
    d_model=384,
    n_layers=4,
    nhead=6,
    dropout=0.2,
    max_len=1024,
    num_classes=4,
    d_state=16,
    d_conv=4,
    expand=2,
    use_front_conv=True,
):
    model = SSA_HybridTokensModel(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        use_front_conv=use_front_conv,
    )
    return model


# ============================================================
# ✅ 兼容旧导入名
# ============================================================
SSA_Model = SSA_HybridTokensModel
build_backbone = build_model