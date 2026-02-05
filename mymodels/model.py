import torch
import torch.nn as nn

# =============== RMSNorm 兼容 ===============
def _rmsnorm(dim: int):
    return nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)

# =============== mamba 依赖 ===============
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("⚠️ 未安装 mamba_ssm。需要安装：pip install mamba-ssm causal-conv1d")

# =========================
# 1) 位置编码
# =========================
def sinusoidal_positional_encoding(seq_len: int, dim: int, device="cpu"):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# =========================
# 2) BiMambaBlock（双向 + FFN）
# =========================
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise RuntimeError(
                "当前环境没有安装 mamba_ssm，无法使用 BiMambaBlock。\n"
                "请安装：pip install mamba-ssm causal-conv1d"
            )

        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        return x + self.mlp(self.ln2(x))

# =========================
# 3) AttentionBlock（残差）
# =========================
class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        self.ln = _rmsnorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_n = self.ln(x)
        # 这里不额外 masked_fill，key_padding_mask 已经足够屏蔽 PAD
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        return x + self.drop(x_a)

# =========================
# 4) FFNBlock（残差）
# =========================
class FFNBlock(nn.Module):
    def __init__(self, d_model, dropout=0.3, mult=4):
        super().__init__()
        self.ln = _rmsnorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.ln(x))

# =========================
# 5) ICBHI Pooling：Attn pooling + Max pooling
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
        attn_scores = self.attn_net(x)  # (B,T,1)
        if mask is not None:
           # attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
            if mask is not None:
                mask = mask.to(torch.bool)  # (B, T)

                # 兼容 attn_scores 为 (B,T) 或 (B,T,1)/(B,T,C)
                if attn_scores.dim() == 2:
                    attn_scores = attn_scores.masked_fill(mask, -1e9)
                else:
                    attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)

        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)  # (B,D)

        if mask is not None:
            x_for_max = x.masked_fill(mask.unsqueeze(-1), -1e9)
        else:
            x_for_max = x
        feat_max, _ = torch.max(x_for_max, dim=1)  # (B,D)

        return feat_weighted + feat_max

# =========================
# 6) 一个 Stage（= 1 layer）
#    3×BiMamba -> 1×Attn -> 3×BiMamba -> FFN
# =========================
class Stage3Attn3FFN(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        dropout=0.3,
        d_state=16,
        d_conv=4,
        expand=2,
        ffn_mult=4,
    ):
        super().__init__()

        self.pre_mambas = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(3)
        ])

        self.attn = AttentionBlock(d_model, nhead=nhead, dropout=dropout)

        self.post_mambas = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(3)
        ])

        self.ffn = FFNBlock(d_model, dropout=dropout, mult=ffn_mult)

    def forward(self, x, mask=None):
        for blk in self.pre_mambas:
            x = blk(x)
        x = self.attn(x, mask=mask)
        for blk in self.post_mambas:
            x = blk(x)
        x = self.ffn(x)
        return x

# =========================
# 7) 主模型：PE -> Conv(once) -> [Stage × n_layers] -> Norm -> Pool
# =========================
# =========================
# 修改后的主模型：SSA_Model_HeARTokens
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(
            self,
            in_dim=768,
            d_model=128,  #  改这里
            n_layers=8,
            nhead=4,  #  下面我会说 nhead 怎么配更好
            dropout=0.3,
            max_len=1024,
            num_classes=2,
            conv_k=7,
            d_state=16,
            d_conv=4,
            expand=2,
            ffn_mult=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # --- 修改点 1: 移除降维投影，改为轻量级特征整合 ---
        # 如果你希望完全不改变特征，甚至可以只用 nn.Identity()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            # 仅做线性变换而不改变维度，有助于模型适配后续的 Mamba 结构
            nn.Linear(in_dim, d_model),
            nn.SiLU()
        )

        # --- 修改点 2: 确保 PE 长度覆盖 AST 的 798 个 tokens ---
        pe = sinusoidal_positional_encoding(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        # --- 修改点 3: 卷积层维度同步 ---
        self.front_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_k, padding=conv_k // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

        # --- 修改点 4: Stage 结构会自动继承 d_model=768 ---
        self.stages = nn.ModuleList([
            Stage3Attn3FFN(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ffn_mult=ffn_mult,
            )
            for _ in range(n_layers)
        ])

        self.norm = _rmsnorm(d_model)
        self.pool = ICBHI_Pooling(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 增加一层瓶颈层，平滑分类
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self.token_head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None, return_feature=False):

        x = self.input_proj(x)

        T = x.shape[1]
        # 加上位置编码
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        # 局部特征增强
        x = x + self.front_conv(x.transpose(1, 2)).transpose(1, 2)

        for stage in self.stages:
            x = stage(x, mask=mask)

        x = self.norm(x)
        file_feature = self.pool(x, mask=mask)

        if return_feature:
            return file_feature, x

        file_logits = self.classifier(file_feature)
        #token_logits = self.token_head(x)
        return file_logits

# =========================
# 8) Route-A Backbone：输出 (B,d_model)
# =========================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_HeARTokens):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model

    def forward(self, x, mask=None):
        feat, _ = self.ssa(x, mask=mask, return_feature=True)
        return feat

# =========================
# 9) build_model：返回 backbone（Route-A）
# =========================
def build_model(
    in_dim=768,
    d_model=128,
    n_layers=8,
    nhead=4,
    dropout=0.3,
    max_len=512,
    conv_k=7,
    d_state=16,
    d_conv=4,
    expand=2,
    ffn_mult=4,
):
    ssa = SSA_Model_HeARTokens(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
        num_classes=2,
        conv_k=conv_k,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        ffn_mult=ffn_mult,
    )
    backbone = SSA_Backbone(ssa)

    params = sum(p.numel() for p in backbone.parameters())
    print(f"✅ Structure: PE → Conv(once) → [3×BiMamba → Attn → 3×BiMamba → FFN] × {n_layers}")
    print(f"   Parameters: {params:,} | Feature Dim: {backbone.final_feat_dim}")
    return backbone

# ============================================================
# ✅ 兼容 mymodels/__init__.py
# 你的 __init__.py 写了：from .model import SSA_Model, build_model
# 所以必须提供 SSA_Model 这个名字
# ============================================================
SSA_Model = SSA_Model_HeARTokens

# （可选）同时兼容以前叫 build_backbone 的脚本
build_backbone = build_model
