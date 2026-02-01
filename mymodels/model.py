import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# =========================
# 0) 小工具：sin/cos 位置编码（可变长度更稳）
# =========================
def sinusoidal_positional_encoding(seq_len: int, dim: int, device):
    """
    (seq_len, dim)
    """
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0, device=device)) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =====================================================
# 1) BiMambaBlock: 双向 Mamba 核心块
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
        # x: (B, T, D)
        h = self.ln1(x)
        # 双向：前向 + 翻转的后向
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        x = x + self.mlp(self.ln2(x))
        return x


# =====================================================
# 2) SSA_Layer: 卷积 + (Mamba堆叠) + Attention + (Mamba堆叠) + gate
# =====================================================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3, n_mamba_pre=3, n_mamba_post=3):
        super().__init__()

        # Conv1d expects (B, D, T)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        self.pre_mamba = nn.ModuleList([BiMambaBlock(d_model, dropout=dropout) for _ in range(n_mamba_pre)])

        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.post_mamba = nn.ModuleList([BiMambaBlock(d_model, dropout=dropout) for _ in range(n_mamba_post)])

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, T, D)
        key_padding_mask: (B, T) bool, True means PAD（不参与attention）
        """
        res_layer = x

        # 1) Conv (local)
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x = x + self.conv(x_conv).transpose(1, 2)

        # 2) Pre Mamba
        for blk in self.pre_mamba:
            x = blk(x)

        # 3) Attention (global) - 支持 mask
        res_attn = x
        x_norm = self.attn_ln(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False)
        x_out = res_attn + self.attn_drop(x_attn)

        # 4) Post Mamba
        for blk in self.post_mamba:
            x_out = blk(x_out)

        # 5) Gate（用均值做门控；若有mask则做masked mean）
        if key_padding_mask is None:
            mean_feat = x_out.mean(dim=1, keepdim=True)  # (B,1,D)
        else:
            # mask: True=PAD -> weight=0
            valid = (~key_padding_mask).float().unsqueeze(-1)  # (B,T,1)
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_feat = (x_out * valid).sum(dim=1, keepdim=True) / denom

        g = self.gate(mean_feat)  # (B,1,D)
        return res_layer + g * x_out


# =====================================================
# 3) PatchEncoder: (200,48) -> (D)
#    48 升到 d_model
# =====================================================
class PatchEncoder(nn.Module):
    def __init__(self, in_dim=48, d_model=256, dropout=0.2):
        super().__init__()
        # 输入 patch: (B,T,200,48)
        # 我们把每个 patch 的 200 帧当序列，用 1D conv 提取，再池化成一个向量
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_patch):
        """
        x_patch: (B, T, 200, 48)
        return:  x_token (B, T, D)
        """
        B, T, L, C = x_patch.shape  # L=200, C=48
        # 先对最后一维做升维：48 -> D
        x = self.proj(x_patch)  # (B,T,200,D)

        # 对每个 patch 的 200 帧做 temporal conv + pool 成一个向量
        x = x.reshape(B * T, L, -1).transpose(1, 2)  # (B*T, D, 200)
        x = self.temporal_conv(x)                   # (B*T, D, 200)

        # 池化到 1：得到 patch-level token
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B*T, D)
        x = x.reshape(B, T, -1)                      # (B, T, D)
        return x


# =====================================================
# 4) SSA_Model: 输出 patch logits (B,T)
# =====================================================
class SSA_PatchLogitModel(nn.Module):
    def __init__(
        self,
        in_dim=48,
        d_model=256,
        dropout=0.2,
        n_layers=2,
        nhead=8,
        max_len=256,   # 位置编码最大支持的 T（够用就行）
    ):
        super().__init__()
        self.patch_encoder = PatchEncoder(in_dim=in_dim, d_model=d_model, dropout=dropout)

        self.max_len = max_len
        self.pos_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 输出每个 patch 的 logits
        self.patch_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)  # -> logit
        )

    def forward(self, x_patch, patch_mask=None, return_emb=False):
        """
        x_patch: (B, T, 200, 48)
        patch_mask: (B, T) bool, True=PAD (可选，T可变时用)
        return:
            logits_patch: (B, T)
            (optional) emb: (B, T, D)
        """

        device = x_patch.device
        B, T, _, _ = x_patch.shape

        # 1) patch -> token
        x = self.patch_encoder(x_patch)  # (B, T, D)

        # 2) 位置编码（sin/cos） + dropout
        if T > self.max_len:
            # 允许更长：动态生成
            pos = sinusoidal_positional_encoding(T, x.size(-1), device).unsqueeze(0)  # (1,T,D)
        else:
            pos = sinusoidal_positional_encoding(T, x.size(-1), device).unsqueeze(0)
        x = self.pos_drop(x + pos)

        # print("DEBUG patch_mask type:", type(patch_mask))
        # if patch_mask is not None:
        #     print("DEBUG patch_mask is_tensor:", torch.is_tensor(patch_mask))

        # 3) SSA layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=patch_mask)

        x = self.norm(x)

        # 4) patch logits
        logits = self.patch_head(x).squeeze(-1)  # (B,T)

        # 5) 若有mask，把PAD位置 logits 置为极小（防止你后面 max/topk 选到PAD）
        if patch_mask is not None:
            logits = logits.masked_fill(patch_mask, -1e9)

        if return_emb:
            return logits, x
        return logits



def build_model(in_dim=48, d_model=512, dropout=0.2, n_layers=4, nhead=8, max_len=512):
    """
    你的 train.py 可以:
        model = build_model(in_dim=48, d_model=256, ...)
        logits_patch = model(x_patch, patch_mask=mask)  # (B,T)
        prob_patch = torch.sigmoid(logits_patch)
        file_prob = prob_patch.max(dim=1)[0]            # MIL聚合在train里做
    """
    return SSA_PatchLogitModel(
        in_dim=in_dim,
        d_model=d_model,
        dropout=dropout,
        n_layers=n_layers,
        nhead=nhead,
        max_len=max_len
    )
