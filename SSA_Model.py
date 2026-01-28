import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm causal-conv1d)")
    exit()


# =====================================================
# 1) HeavyBlock: 3路双向 Mamba + 1路 Attention + 1路 MLP
# =====================================================
class HeavyBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # 3组双向 Mamba
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'ln': nn.LayerNorm(d_model),
                'fwd': Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2),
                'bwd': Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
            }) for _ in range(3)
        ])
        self.drop = nn.Dropout(dropout)

        # 1组 Multi-head Attention
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
        # 1. 依次运行 3 层双向 Mamba
        for m in self.mamba_layers:
            # 双向扫描逻辑
            fwd_out = m['fwd'](m['ln'](x))
            bwd_out = torch.flip(m['bwd'](torch.flip(m['ln'](x), [1])), [1])
            x = x + self.drop(fwd_out + bwd_out)

        # 2. Attention 层
        res = x
        x_a, _ = self.attn(self.ln_a(x), self.ln_a(x), self.ln_a(x))
        x = res + self.drop(x_a)

        # 3. MLP 层
        x = x + self.mlp(self.ln_mlp(x))
        return x


# =====================================================
# 2) SSA_Model: 接收 HeAR Embedding 的纯净架构
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=768, num_classes=1, n_layers=6, d_model=192, dropout=0.2):
        super().__init__()

        # A. 输入投影：对齐 HeAR 维度并应用你要求的 0.2 Dropout
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(0.2)

        # B. 统一使用 HeavyBlock
        self.blocks = nn.ModuleList([
            HeavyBlock(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 注意力池化：让模型自动决定哪些 Patch 更重要
        self.pool_proj = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x 形状: [Batch, Time, 768] (来自 HeAR Patch Embedding)

        # 1. 投影与初始 Dropout
        x = self.input_proj(x)
        x = self.input_dropout(x)

        # 2. 经过 HeavyBlock 堆叠
        for block in self.blocks:
            x = block(x)

        # 3. 全局规范化与注意力池化
        x = self.norm(x)
        weights = torch.softmax(self.pool_proj(x), dim=1)
        feat = torch.sum(x * weights, dim=1)

        # 4. 分类头输出
        return self.head(feat).squeeze(-1)