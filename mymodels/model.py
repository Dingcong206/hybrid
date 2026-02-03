import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. 辅助模块：通道注意力 (SE Block)
# =========================
class SE_Block(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, D)
        b, t, d = x.size()
        y = torch.mean(x, dim=1)  # 全局时间平均
        y = self.fc(y).view(b, 1, d)
        return x * y.expand_as(x)


# =========================
# 2. 改进后的 SSA 层
# =========================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        # 局部纹理提取：深度卷积
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

        # 序列建模：双向 Mamba
        self.mamba_pre = BiMambaBlock(d_model, dropout=dropout)

        # 全局修正：Attention
        self.attn_ln = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 通道筛选：SE Block (ICBHI 提分关键)
        self.se = SE_Block(d_model)

    def forward(self, x, mask=None):
        # 1. 局部卷积残差
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # 2. Mamba 扫描
        x = self.mamba_pre(x)

        # 3. Attention
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        x = x + x_a

        # 4. 通道注意力增强
        x = self.se(x)
        return x


# =========================
# 3. 层级注意力池化 (Patient/File Level)
# =========================
class ICBHI_Pooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        # 路径 A: 时间权重注意力 (捕捉关键帧)
        attn_scores = self.attn_net(x)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)

        # 路径 B: 全局最大池化 (捕捉爆裂音瞬间特征)
        # 屏蔽 padding 部分的影响
        if mask is not None:
            x_for_max = x.masked_fill(mask.unsqueeze(-1), -1e9)
        else:
            x_for_max = x
        feat_max, _ = torch.max(x_for_max, dim=1)

        # 融合两条路径
        return feat_weighted + feat_max


# =========================
# 4. 最终集成模型
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(
            self,
            in_dim=768,
            d_model=256,
            n_layers=4,
            nhead=8,
            dropout=0.3,
            max_len=4096,
            num_classes=4,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            nn.RMSNorm(d_model),
        )

        # 2. 位置编码
        pe = sinusoidal_positional_encoding(max_len, d_model, device='cpu')
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        # 3. 堆叠改进后的 SSA 层
        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.RMSNorm(d_model)

        # 4. ICBHI 专用层级池化
        self.pool = ICBHI_Pooling(d_model)

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        # (可选) Token 级别预测头
        self.token_head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None, return_feature=False):
        # 投影与位置增强
        x = self.input_proj(x)
        T = x.shape[1]
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        # SSA 深度特征提取
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)

        # 层级注意力池化 (Token 200 -> 1 Feature)
        file_feature = self.pool(x, mask=mask)

        if return_feature:
            return file_feature, x

        # 输出
        file_logits = self.classifier(file_feature)
        token_logits = self.token_head(x)
        return file_logits, token_logits


# =========================
# 5. Backbone 封装
# =========================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_HeARTokens):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model

    def forward(self, x, mask=None):
        # 直接返回池化后的病人/录音级特征
        feat, _ = self.ssa(x, mask=mask, return_feature=True)
        return feat