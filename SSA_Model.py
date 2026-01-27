import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm causal-conv1d)")
    exit()


# =====================================================
# 1) ConvStem (保持和你代码一模一样)
# =====================================================
class ConvStem(nn.Module):
    def __init__(self, embed_dim=192, patch_time=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        stride_w = max(1, patch_time // 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=(64, 3), stride=(64, stride_w), padding=(0, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.squeeze(2).transpose(1, 2)
        return self.norm(x)


# =====================================================
# 2) SSABlock (现在是 3个 Bi-Mamba + 1个 Attention)
# =====================================================
class SSABlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # Mamba 1
        self.ln1 = nn.LayerNorm(d_model)
        self.m1_fwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.m1_bwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.drop1 = nn.Dropout(dropout)

        # Mamba 2 (新增)
        self.ln2 = nn.LayerNorm(d_model)
        self.m2_fwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.m2_bwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.drop2 = nn.Dropout(dropout)

        # Mamba 3 (新增)
        self.ln3 = nn.LayerNorm(d_model)
        self.m3_fwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.m3_bwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.drop3 = nn.Dropout(dropout)

        # Attention 支路
        self.ln_a = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop_a = nn.Dropout(dropout)

        # MLP 支路
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Mamba 1
        x = x + self.drop1(self.m1_fwd(self.ln1(x)) + torch.flip(self.m1_bwd(torch.flip(self.ln1(x), [1])), [1]))

        # Mamba 2
        x = x + self.drop2(self.m2_fwd(self.ln2(x)) + torch.flip(self.m2_bwd(torch.flip(self.ln2(x), [1])), [1]))

        # Mamba 3
        x = x + self.drop3(self.m3_fwd(self.ln3(x)) + torch.flip(self.m3_bwd(torch.flip(self.ln3(x), [1])), [1]))

        # Attention
        res = x
        x_a, _ = self.attn(self.ln_a(x), self.ln_a(x), self.ln_a(x))
        x = res + self.drop_a(x_a)

        # MLP
        x = x + self.mlp(self.ln_mlp(x))
        return x


# =====================================================
# 3) SSA_Model 整体架构
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, num_classes=1, n_layers=6, d_model=192, patch_time=4, dropout=0.3):
        super().__init__()
        self.stem = ConvStem(embed_dim=d_model, patch_time=patch_time)

        # 保持和你代码一致的 pos_embed 初始化
        num_patches = 1024 // patch_time
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        self.blocks = nn.ModuleList([
            SSABlock(d_model=d_model, dropout=dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        B, L, D = x.shape
        x = x + self.pos_embed[:, :L, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        weights = torch.softmax(self.pool_proj(x), dim=1)
        feat = torch.sum(x * weights, dim=1)
        return self.head(feat).squeeze(-1)