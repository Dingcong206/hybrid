# VimA_Model.py
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("请先安装: pip install mamba-ssm causal-conv1d")


# =====================================================
# 1) Stem：声学条带卷积，把 [B,1,128,1024] -> [B,L,D]
# =====================================================
class AcousticStripStem(nn.Module):
    def __init__(self, freq_bins=128, patch_time=4, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=(freq_bins, patch_time),
            stride=(freq_bins, patch_time)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 1, 128, 1024]
        x = self.proj(x)                  # [B, D, 1, L]
        x = x.flatten(2).transpose(1, 2)  # [B, L, D]
        return self.norm(x)


# =====================================================
# 2) SSM 子层：Bi-Mamba + MLP（不含 attention）
# =====================================================
class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.ln_m = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.drop_m = nn.Dropout(dropout)

        self.ln_ff = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop_ff = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        # ---- Bi-Mamba ----
        res = x
        x1 = self.ln_m(x)
        out_fwd = self.mamba_fwd(x1)
        out_bwd = torch.flip(self.mamba_bwd(torch.flip(x1, dims=[1])), dims=[1])
        x = res + self.drop_m(out_fwd + out_bwd)

        # ---- FFN ----
        res = x
        x2 = self.ln_ff(x)
        x = res + self.drop_ff(self.mlp(x2))
        return x


# =====================================================
# 3) Attention 子层：MHA + MLP（不含 mamba）
# =====================================================
class AttnBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.ln_a = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.drop_a = nn.Dropout(dropout)

        self.ln_ff = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop_ff = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        res = x
        q = self.ln_a(x)
        a, _ = self.attn(q, q, q, need_weights=False)
        x = res + self.drop_a(a)

        res = x
        x2 = self.ln_ff(x)
        x = res + self.drop_ff(self.mlp(x2))
        return x


# =====================================================
# 4) 你要的 macro-layer：SSM×3 + Attn×1
# =====================================================
class MacroLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.ssm1 = SSMBlock(d_model=d_model, dropout=dropout)
        self.ssm2 = SSMBlock(d_model=d_model, dropout=dropout)
        self.ssm3 = SSMBlock(d_model=d_model, dropout=dropout)
        self.attn = AttnBlock(d_model=d_model, nhead=nhead, dropout=dropout)

    def forward(self, x):
        x = self.ssm1(x)
        x = self.ssm2(x)
        x = self.ssm3(x)
        x = self.attn(x)
        return x


# =====================================================
# 5) 整体模型： (SSM×3 + Attn×1) × 6
#    + Attention Pooling 做分类
# =====================================================
class VimAHybrid(nn.Module):
    def __init__(
        self,
        num_classes=1,
        d_model=192,
        patch_time=4,
        num_layers=6,        # ✅ 你要的 6 个 macro-layer
        nhead=8,
        dropout=0.1
    ):
        super().__init__()
        self.stem = AcousticStripStem(freq_bins=128, patch_time=patch_time, embed_dim=d_model)

        num_patches = 1024 // patch_time
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        self.layers = nn.ModuleList([
            MacroLayer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # ✅ Attention Pooling（对所有 patch tokens 加权求和）
        self.attn_pool = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B,1,128,1024]
        x = self.stem(x)              # [B,L,D]
        x = x + self.pos_embed        # [B,L,D]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)              # [B,L,D]

        # Attention Pool
        attn_score = self.attn_pool(x)                 # [B,L,1]
        attn_weight = torch.softmax(attn_score, dim=1) # [B,L,1]
        feat = (x * attn_weight).sum(dim=1)            # [B,D]

        return self.head(feat).squeeze(-1)             # [B]


if __name__ == "__main__":
    # quick test
    model = VimAHybrid(num_layers=6, patch_time=4, d_model=192, nhead=8)
    x = torch.randn(2, 1, 128, 1024)
    y = model(x)
    print("OK:", y.shape)  # [2]
