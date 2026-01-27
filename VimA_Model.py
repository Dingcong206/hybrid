import torch
import torch.nn as nn

# 尝试导入 Mamba 内核，如果没装，会报错提示
try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误: 请先安装 mamba-ssm (pip install mamba-ssm causal-conv1d)")
    exit() # 如果没有安装，直接退出，避免后续错误

# =====================================================
# 第一部分：声学条带卷积 (Stem)
# 将 (128, 1024) 的频谱切成一个个“时间窄、频率宽”的条带
# =====================================================
class AcousticStripStem(nn.Module):
    def __init__(self, freq_bins=128, patch_time=4, embed_dim=192):
        super().__init__()
        # 卷积核高度=128(全频率), 宽度=patch_time(窄时间)
        # stride=(freq_bins, patch_time) 确保不重叠地切分
        self.proj = nn.Conv2d(
            1, embed_dim,
            kernel_size=(freq_bins, patch_time),
            stride=(freq_bins, patch_time)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 1, 128, 1024]
        x = self.proj(x)  # -> [B, embed_dim, 1, L_patches]  (L_patches = 1024 / patch_time)
        x = x.flatten(2).transpose(1, 2)  # -> [B, L_patches, embed_dim]
        return self.norm(x)


# =====================================================
# 第二部分：混合 Mamba-Attention 块 (Inspired by Vim & Hybrid Logic)
# 结合了 Mamba 的序列效率和 Attention 的全局标定
# =====================================================
class HybridBlock(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        # 1. 双向 Mamba 支路
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

        # 2. 自注意力支路 (用于锁定瞬时病理特征)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 3. 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout_mamba = nn.Dropout(0.1) # 增加 dropout_mamba
        self.dropout_attn = nn.Dropout(0.1)  # 增加 dropout_attn
        self.dropout_mlp = nn.Dropout(0.1)   # 增加 dropout_mlp

    def forward(self, x):
        # Mamba 扫描
        res = x
        x1 = self.ln1(x)
        # 正向 + 反向 (Flip 技巧实现双向)
        x_m = self.mamba_fwd(x1) + torch.flip(self.mamba_bwd(torch.flip(x1, [1])), [1])
        x = res + self.dropout_mamba(x_m) # 应用 dropout

        # Attention 标定
        res = x
        x_a, _ = self.attn(self.ln2(x), self.ln2(x), self.ln2(x))
        x = res + self.dropout_attn(x_a) # 应用 dropout

        # MLP
        x = x + self.dropout_mlp(self.mlp(x)) # 应用 dropout
        return x


# =====================================================
# 第三部分：整机架构 (VimA-Hybrid) - 引入 CLS Token
# =====================================================
class VimAHybrid(nn.Module):
    def __init__(self, num_classes=1, n_layers=8, d_model=192, patch_time=4):
        super().__init__()
        # 1. 卷积前端
        self.stem = AcousticStripStem(freq_bins=128, patch_time=patch_time, embed_dim=d_model)

        # 2. 可学习的 CLS Token 和位置编码
        num_patches = 1024 // patch_time
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) # [1, 1, D]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model)) # 包含 CLS Token 的位置编码

        # 3. 堆叠混合层
        self.blocks = nn.ModuleList([
            HybridBlock(d_model=d_model) for _ in range(n_layers)
        ])

        # 4. 输出头
        self.norm = nn.LayerNorm(d_model)
        # Attention Pooling
        self.attn_pool = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, num_classes)

        # 初始化权重 (可选，有助于稳定训练)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, 1, 128, 1024]
        x = self.stem(x)  # -> [B, L_patches, embed_dim]

        # 拼接 CLS Token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # 扩展到 Batch size
        x = torch.cat((cls_tokens, x), dim=1) # -> [B, L_patches + 1, embed_dim]

        x = x + self.pos_embed # 添加位置编码

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # 关键：只使用 CLS Token 的输出进行分类
       # return self.head(x[:, 0]).squeeze(-1) # -> [B]
        # 去掉 CLS，只用 patch tokens
        x = x[:, 1:]  # [B, L, D]

        # 计算注意力权重
        attn_score = self.attn_pool(x)  # [B, L, 1]
        attn_weight = torch.softmax(attn_score, dim=1)

        # 加权求和
        feat = (x * attn_weight).sum(dim=1)  # [B, D]

        return self.head(feat).squeeze(-1)
