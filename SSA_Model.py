import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm causal-conv1d)")
    exit()


# =====================================================
# 1) ConvStem：3x3 卷积堆叠提取局部声学特征
# =====================================================
class ConvStem(nn.Module):
    def __init__(self, embed_dim=192, patch_time=4):
        super().__init__()
        # 第一层：局部纹理提取 [B, 1, 128, 1024] -> [B, 64, 64, 512]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # 第二层：频率下采样至1，时间轴进一步下采样
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
        x = x.squeeze(2).transpose(1, 2)  # [B, L, D]
        return self.norm(x)


# =====================================================
# 2) SSABlock：双 Bi-Mamba + 单 Attention 混合块
# =====================================================
class SSABlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # 第一组双向 Mamba：捕捉基础呼吸节律
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba1_fwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.mamba1_bwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.drop1 = nn.Dropout(dropout)

        # 第二组双向 Mamba：强化时序特征过滤
        self.ln2 = nn.LayerNorm(d_model)
        self.mamba2_fwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.mamba2_bwd = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        self.drop2 = nn.Dropout(dropout)

        # 自注意力层：锁定突发的病理音（如爆裂音）
        self.ln3 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop_a = nn.Dropout(dropout)

        # 前馈网络：特征非线性变换
        self.ln4 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Mamba 1
        res = x
        x1 = self.ln1(x)
        x_m1 = self.mamba1_fwd(x1) + torch.flip(self.mamba1_bwd(torch.flip(x1, [1])), [1])
        x = res + self.drop1(x_m1)

        # Mamba 2
        res = x
        x2 = self.ln2(x)
        x_m2 = self.mamba2_fwd(x2) + torch.flip(self.mamba2_bwd(torch.flip(x2, [1])), [1])
        x = res + self.drop2(x_m2)

        # Attention
        res = x
        x_a, _ = self.attn(self.ln3(x), self.ln3(x), self.ln3(x))
        x = res + self.drop_a(x_a)

        # MLP
        x = x + self.mlp(self.ln4(x))
        return x


# =====================================================
# 3) SSA_Model 整体架构
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, num_classes=1, n_layers=4, d_model=192, patch_time=4, dropout=0.3):
        super().__init__()
        # 因为每一层 SSABlock 都很厚，建议 n_layers 设为 4-6
        self.stem = ConvStem(embed_dim=d_model, patch_time=patch_time)

        # 位置编码
        num_patches = 1024 // patch_time
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        # 堆叠混合块
        self.blocks = nn.ModuleList([
            SSABlock(d_model=d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Attention Pooling (比 CLS Token 更有利于寻找病理片段)
        self.pool_proj = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, num_classes)

        # 初始化权重
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
        # x: [B, 1, 128, 1024]
        x = self.stem(x)  # [B, L, D]

        # 位置编码对齐
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Attention Pooling：模型会自动给有病理音的时间段分配更高权重
        weights = torch.softmax(self.pool_proj(x), dim=1)  # [B, L, 1]
        feat = torch.sum(x * weights, dim=1)  # [B, D]

        return self.head(feat).squeeze(-1)  # [B]


