import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm causal-conv1d)")
    exit()


# =====================================================
# 1) ConvStem (保持逻辑一致)
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
# 2) 轻量块 (LightBlock)：用于底层
# =====================================================
class LightBlock(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.m_fwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.m_bwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # 仅一层双向 Mamba + MLP
        x = x + self.drop(self.m_fwd(self.ln(x)) + torch.flip(self.m_bwd(torch.flip(self.ln(x), [1])), [1]))
        x = x + self.mlp(self.ln_mlp(x))
        return x


# =====================================================
# 3) 超厚块 (HeavyBlock)：用于顶层 (3-Mamba + 1-Attention)
# =====================================================
class HeavyBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # 3组 Mamba
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'ln': nn.LayerNorm(d_model),
                'fwd': Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2),
                'bwd': Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
            }) for _ in range(3)
        ])
        self.drop = nn.Dropout(dropout)

        # 1组 Attention
        self.ln_a = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 1组 MLP
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # 依次运行 3 层双向 Mamba
        for m in self.mamba_layers:
            x = x + self.drop(m['fwd'](m['ln'](x)) + torch.flip(m['bwd'](torch.flip(m['ln'](x), [1])), [1]))

        # Attention
        res = x
        x_a, _ = self.attn(self.ln_a(x), self.ln_a(x), self.ln_a(x))
        x = res + self.drop(x_a)

        # MLP
        x = x + self.mlp(self.ln_mlp(x))
        return x


# =====================================================
# 4) SSA_Model (非对称架构)
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, num_classes=1, n_layers=8, d_model=192, patch_time=2, dropout=0.3):
        super().__init__()
        self.stem = ConvStem(embed_dim=d_model, patch_time=patch_time)

        num_patches = 1024 // patch_time
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # 核心修改：分层堆叠
        self.blocks = nn.ModuleList()
        mid = n_layers // 2
        for i in range(n_layers):
            if i < mid:
                # 前半部分：轻量层
                self.blocks.append(LightBlock(d_model, dropout))
            else:
                # 后半部分：厚重层
                self.blocks.append(HeavyBlock(d_model, nhead=8, dropout=dropout))

        self.norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(0.2)
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
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        weights = torch.softmax(self.pool_proj(x), dim=1)
        feat = torch.sum(x * weights, dim=1)
        return self.head(feat).squeeze(-1)