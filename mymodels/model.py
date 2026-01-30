import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# =====================================================
# 1) BiMambaBlock: 双向 Mamba 核心块 (保持不变)
# =====================================================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        x = x + self.mlp(self.ln2(x))
        return x

# =====================================================
# 2) SSA_Layer: 每一层 = Attention -> 3*Mamba -> Attention
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # A. 卷积层：局部细节补偿
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # B. 前置 Attention 层
        self.pre_attn_ln = nn.LayerNorm(d_model)
        self.pre_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.pre_attn_drop = nn.Dropout(dropout)

        # C. 中间 3 层 Mamba 堆栈
        self.mamba_stack = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout)
            for _ in range(3)
        ])

        # D. 后置 Attention 层
        self.post_attn_ln = nn.LayerNorm(d_model)
        self.post_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.post_attn_drop = nn.Dropout(dropout)

        # E. 门控残差机制
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_layer = x  # 这一层的大残差

        # 1. 卷积局部增强
        x_conv = x.transpose(1, 2)
        x = x + self.conv(x_conv).transpose(1, 2)

        # 2. 前置 Attention：先看全局
        res_pre = x
        x_pre, _ = self.pre_attn(self.pre_attn_ln(x), self.pre_attn_ln(x), self.pre_attn_ln(x))
        x = res_pre + self.pre_attn_drop(x_pre)

        # 3. 3 层 Mamba：深度时序建模
        for mamba in self.mamba_stack:
            x = mamba(x)

        # 4. 后置 Attention：全局特征归纳
        res_post = x
        x_post, _ = self.post_attn(self.post_attn_ln(x), self.post_attn_ln(x), self.post_attn_ln(x))
        x = res_post + self.post_attn_drop(x_post)

        # 5. 门控融合
        g = self.gate(x.mean(dim=1, keepdim=True))
        return res_layer + g * x

# =====================================================
# 3) SSA_Model: 顶层封装
# =====================================================
class SSA_Model(nn.Module):
    def __init__(self, input_dim=1024, num_classes=1, d_model=256, dropout=0.2, n_layers=2, seq_len=96):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.pos_drop = nn.Dropout(dropout)

        # 保持 SSA_Layer 不变
        self.layers = nn.ModuleList([
            SSA_Layer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 为每一层的输出准备一个 Norm，防止浅层和深层数值差距过大
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # 池化组件
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        # 分类头：因为拼接了 n_layers 层，所以输入维度要乘以 n_layers
        # 且原本就有 3 路池化 (Attn, Max, Avg)，所以是 d_model * 3 * n_layers
        self.head = nn.Sequential(
            nn.Linear(d_model * 3 * n_layers, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.pos_drop(x)

        all_layer_features = []

        # 1. 逐层前传，并收集每一层的特征
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 对当前层输出做池化前处理
            layer_output = self.layer_norms[i](x)

            # 2. 对每一层分别进行 Top-K + Max + Avg 池化
            # (这样能捕捉到每一层认为“最重要”的时刻)
            attn_logits = self.pool_proj(layer_output)
            _, topk_idx = torch.topk(attn_logits.squeeze(-1), k=16, dim=1)
            mask = torch.zeros_like(attn_logits).fill_(-1e9)
            mask.scatter_(1, topk_idx.unsqueeze(-1), 0)
            topk_weights = torch.softmax((attn_logits + mask) / 0.5, dim=1)

            attn_f = torch.sum(layer_output * topk_weights, dim=1)
            max_f, _ = torch.max(layer_output, dim=1)
            avg_f = torch.mean(layer_output, dim=1)

            # 每一层贡献的特征向量：[B, d_model * 3]
            combined_f = torch.cat([attn_f, max_f, avg_f], dim=-1)
            all_layer_features.append(combined_f)

        # 3. 将所有层的特征拼接在一起：[B, d_model * 3 * n_layers]
        final_feat = torch.cat(all_layer_features, dim=-1)

        return self.head(final_feat).squeeze(-1)

def build_model(input_dim=1024, d_model=256, dropout=0.15, num_classes=1, seq_len=96):
    return SSA_Model(input_dim, num_classes, d_model, dropout, n_layers=2, seq_len=seq_len)