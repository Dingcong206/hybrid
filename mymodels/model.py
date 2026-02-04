import torch
import torch.nn as nn
from mamba_ssm import Mamba


# ==========================================
# 1. 位置编码 (Sinusoidal Positional Encoding)
# ==========================================
def sinusoidal_positional_encoding(seq_len: int, dim: int, device):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ==========================================
# 2. 核心基础组件
# ==========================================
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BiMambaBlock(nn.Module):
    """双向 Mamba 单元"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.ln(x)
        x_fwd = self.fwd(x)
        # 翻转序列实现反向扫描
        x_bwd = torch.flip(self.bwd(torch.flip(x, [1])), [1])
        return res + self.drop(x_fwd + x_bwd)


# ==========================================
# 3. 复合层设计 (3-1-3 结构)
# ==========================================
class DeepSSALayer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()

        # A. 基层卷积：局部特征融合
        self.pre_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # B. 第一组 3个 BiMamba
        self.mamba_group1 = nn.Sequential(*[BiMambaBlock(d_model, dropout) for _ in range(3)])
        self.ffn1_ln = nn.LayerNorm(d_model)
        self.ffn1 = FeedForward(d_model, dropout)

        # C. 中间 Attention 扫描
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # D. 第二组 3个 BiMamba
        self.mamba_group2 = nn.Sequential(*[BiMambaBlock(d_model, dropout) for _ in range(3)])
        self.ffn2_ln = nn.LayerNorm(d_model)
        self.ffn2 = FeedForward(d_model, dropout)

    def forward(self, x, mask=None):
        # 1. Conv 局部融合
        res = x
        x = x.transpose(1, 2)
        x = self.pre_conv(x).transpose(1, 2)
        x = x + res

        # 2. Mamba 组 1 + FFN
        x = self.mamba_group1(x)
        x = x + self.ffn1(self.ffn1_ln(x))

        # 3. 全局 Attention
        res = x
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        x = res + x_a

        # 4. Mamba 组 2 + FFN
        x = self.mamba_group2(x)
        x = x + self.ffn2(self.ffn2_ln(x))

        return x


# ==========================================
# 4. 主架构 (12 Layers)
# ==========================================
class SSA_Heavy_12Layer(nn.Module):
    def __init__(self, in_dim=768, d_model=256, n_layers=12, nhead=8, dropout=0.1):
        super().__init__()

        # 初始特征映射
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 堆叠 12 个复合层
        self.layers = nn.ModuleList([
            DeepSSALayer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # MIL (多示例学习) 聚合池化层
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        self.classifier = nn.Linear(d_model, 1)  # 文件级输出
        self.patch_head = nn.Linear(d_model, 1)  # Patch级输出

    def forward(self, x, mask=None):
        # x 形状: (B, L, in_dim)
        x = self.input_proj(x)

        # 位置编码
        B, L, D = x.shape
        x = x + sinusoidal_positional_encoding(L, D, x.device).unsqueeze(0)

        # 12 层前向传播
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        # Attention Pooling 逻辑
        attn_scores = self.attention_pool(x)  # (B, L, 1)
        if mask is not None:
            # 排除 padding 的部分
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)

        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L, 1)

        # 聚合得到全局特征
        global_feat = torch.sum(attn_weights * x, dim=1)  # (B, D)

        # 输出 Logits
        file_logit = self.classifier(global_feat).squeeze(-1)  # (B,)
        token_logits = self.patch_head(x).squeeze(-1)  # (B, L)

        return file_logit, token_logits


# ==========================================
# 5. 构造函数
# ==========================================
def build_model(in_dim=256, d_model=256, n_layers=12, nhead=8):
    """
    针对 24GB 显存设计的重型 Mamba-Attention 混合架构
    """
    model = SSA_Heavy_12Layer(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 成功初始化 Heavy SSA 模型 (12层 3-1-3 架构)")
    print(f"📊 总参数量: {total_params:,}")
    return model