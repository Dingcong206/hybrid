# mymodels/model.py
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception as e:
    Mamba = None


# -------------------------
# 基础：FFN
# -------------------------
class FFN(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# BiMambaBlock（带残差）
# -------------------------
class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state=16, d_conv=4, expand=2, dropout=0.1, conv_k: int = 7, ffn_mult: int = 4):
        super().__init__()
        if Mamba is None:
            raise RuntimeError("mamba_ssm 未安装：pip install mamba-ssm causal-conv1d")

        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mult=ffn_mult, dropout=dropout)

    def forward(self, x):
        # x: (B,T,D)
        h = self.ln1(x)
        y_f = self.fwd(h)
        y_b = torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])
        y = y_f + y_b
        x = x + self.drop(y)

        x = x + self.ffn(self.ln2(x))
        return x


# -------------------------
# Attention Block（1层）
# -------------------------
class AttnBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        # 用标准 TransformerEncoderLayer 作为“attention + FFN”
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, key_padding_mask: torch.Tensor):
        # key_padding_mask: True=PAD  (和你 collate_pad 对齐)
        return self.layer(x, src_key_padding_mask=key_padding_mask)


# -------------------------
# 3 + 1 + 3 Backbone
# -------------------------
class Hybrid313Backbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        dropout: float = 0.2,
        max_len: int = 4096,
        # mamba
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        conv_k: int = 7,
        ffn_mult: int = 4,
        # attn
        nhead: int = 8,
        pre_layers: int = 3,
        post_layers: int = 3,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.drop = nn.Dropout(dropout)

        self.pre = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                conv_k=conv_k,
                ffn_mult=ffn_mult,
            )
            for _ in range(pre_layers)
        ])

        self.attn = AttnBlock(d_model=d_model, nhead=nhead, dropout=dropout, ffn_mult=ffn_mult)

        self.post = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                conv_k=conv_k,
                ffn_mult=ffn_mult,
            )
            for _ in range(post_layers)
        ])

        self.final_feat_dim = d_model

    def forward(self, x, mask: torch.Tensor):
        """
        x: (B,T,in_dim)
        mask: (B,T) True=PAD
        return: (B, d_model)
        """
        x = self.drop(self.in_proj(x))

        for blk in self.pre:
            x = blk(x)

        x = self.attn(x, key_padding_mask=mask)

        for blk in self.post:
            x = blk(x)

        # masked mean pooling
        valid = (~mask).float()  # (B,T)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)
        return pooled


# -------------------------
# 你训练脚本调用的入口：build_backbone
# -------------------------
def build_backbone(
    in_dim: int,
    d_model: int = 256,
    n_layers: int = 2,      # 兼容旧参数：这里不再用
    nhead: int = 8,
    dropout: float = 0.2,
    max_len: int = 4096,
    conv_k: int = 7,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    ffn_mult: int = 4,
    # 新增：固定 3+1+3
    pre_layers: int = 3,
    post_layers: int = 3,
):
    return Hybrid313Backbone(
        in_dim=in_dim,
        d_model=d_model,
        dropout=dropout,
        max_len=max_len,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        conv_k=conv_k,
        ffn_mult=ffn_mult,
        nhead=nhead,
        pre_layers=pre_layers,
        post_layers=post_layers,
    )
