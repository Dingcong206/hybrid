import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm causal-conv1d)")
    exit()


# =====================================================
# 1) HeavyBlock: 针对 1024 维特征优化的混合块
# =====================================================
class HeavyBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # 双向 Mamba 逻辑
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'ln': nn.LayerNorm(d_model),
                # 对于 97 步的长序列，d_state 设为 16 或 32 效果更佳
                'fwd': Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                'bwd': Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            }) for _ in range(3)
        ])
        self.drop = nn.Dropout(dropout)

        self.ln_a = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # x: [B, 97, d_model]
        for m in self.mamba_layers:
            norm_x = m['ln'](x)
            fwd_out = m['fwd'](norm_x)
            # 反向扫描：捕捉呼吸音在时间轴上的倒溯特征
            bwd_out = torch.flip(m['bwd'](torch.flip(norm_x, [1])), [1])
            x = x + self.drop(fwd_out + bwd_out)

        res = x
        x_a, _ = self.attn(self.ln_a(x), self.ln_a(x), self.ln_a(x))
        x = res + self.drop(x_a)
        x = x + self.mlp(self.ln_mlp(x))
        return x


# =====================================================
# 2) SSA_Model: 适配 (97, 1024) 维度的架构
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, n_layers=8, d_model=256, dropout=0.2):
        super().__init__()

        # A. 输入投影：将 1024 维降维至模型内部维度 (例如 256)
        # 增加 Dropout 缓解 6898 个样本下的过拟合
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(0.2)

        # B. 堆叠混合块
        self.blocks = nn.ModuleList([
            HeavyBlock(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # C. 分类头：结合全局 CLS 信息和 Patch 聚合信息
        self.pool_proj = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x 形状: [Batch, 97, 1024]

        # 1. 投影
        x = self.input_proj(x)  # [Batch, 97, d_model]
        x = self.input_dropout(x)

        # 2. 核心层计算
        for block in self.blocks:
            x = block(x)

        # 3. 混合池化策略
        x = self.norm(x)

        # 方案：利用 Attention Pooling 自动学习这 97 个位置的重要性
        # (包含 CLS 和所有音频 Patches)
        weights = torch.softmax(self.pool_proj(x), dim=1)
        feat = torch.sum(x * weights, dim=1)  # [Batch, d_model]

        # 4. 输出
        return self.head(feat).squeeze(-1)  # 训练时使用 BCEWithLogitsLoss