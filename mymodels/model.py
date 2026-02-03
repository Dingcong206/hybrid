import torch
import torch.nn as nn

# =============== RMSNorm 兼容（torch 版本不够会没有 nn.RMSNorm） ===============
def _rmsnorm(dim: int):
    return nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)

# =============== mamba 依赖（没有就提示） ===============
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("⚠️ 未安装 mamba_ssm。需要的话安装：pip install mamba-ssm causal-conv1d")

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
# 2) BiMambaBlock（双向）
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
        """
        x: (B,T,D)
        """
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        return x + self.mlp(self.ln2(x))

# =========================
# 3) ICBHI Pooling：Attention pooling + Max pooling
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
        # x: (B,T,D)
        attn_scores = self.attn_net(x)  # (B,T,1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_w = torch.softmax(attn_scores, dim=1)
        feat_weighted = torch.sum(attn_w * x, dim=1)  # (B,D)

        # max pooling（屏蔽 padding）
        if mask is not None:
            x_for_max = x.masked_fill(mask.unsqueeze(-1), -1e9)
        else:
            x_for_max = x
        feat_max, _ = torch.max(x_for_max, dim=1)     # (B,D)

        return feat_weighted + feat_max

# =========================
# 4) 并行层：Conv -> 并行(3×BiMamba + 1×Attn) -> softmax融合 -> FFN
# =========================
class ParallelConv3BiMamba1AttnLayer(nn.Module):
    """
    结构：
      1) Depthwise Conv 残差
      2) Norm
      3) 并行：BiMamba#1/#2/#3 + Attention
      4) Softmax 融合四路输出
      5) 残差 + FFN
    """
    def __init__(self, d_model, nhead=8, dropout=0.3, conv_k=7,
                 mamba_dropout=0.3, d_state=16, d_conv=4, expand=2):
        super().__init__()

        # 0) local depthwise conv
        self.local = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_k, padding=conv_k // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

        # 1) 输入归一化
        self.norm_in = _rmsnorm(d_model)

        # 2) 并行 3 个 BiMamba
        self.bimambas = nn.ModuleList([
            BiMambaBlock(d_model, dropout=mamba_dropout, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(3)
        ])

        # 3) 并行 1 个 Attention
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 4) 融合：输出 (B,T,4) 权重，对应 [b1,b2,b3,a]
        self.fuse_logits = nn.Linear(d_model, 4)
        self.drop = nn.Dropout(dropout)

        # 5) FFN 残差
        self.norm_ffn = _rmsnorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """
        x: (B,T,D)
        mask: (B,T) True=PAD
        """
        # 0) local conv residual
        x = x + self.local(x.transpose(1, 2)).transpose(1, 2)

        # 1) normalize for parallel branches
        x_n = self.norm_in(x)

        # padding token 尽量不影响 mamba/融合
        if mask is not None:
            x_n = x_n.masked_fill(mask.unsqueeze(-1), 0.0)

        # 2) parallel branches
        b_outs = [blk(x_n) for blk in self.bimambas]  # 3*(B,T,D)
        a_out, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)  # (B,T,D)

        outs = torch.stack(b_outs + [a_out], dim=2)  # (B,T,4,D)

        # 3) softmax weights
        w = torch.softmax(self.fuse_logits(x_n), dim=-1)  # (B,T,4)
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1), 0.0)

        mix = (outs * w.unsqueeze(-1)).sum(dim=2)  # (B,T,D)

        # 4) residual
        x = x + self.drop(mix)

        # 5) FFN residual
        x = x + self.ffn(self.norm_ffn(x))
        return x

# =========================
# 5) 主模型（支持 return_feature，路线A用）
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
        conv_k=7,
        mamba_dropout=None,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        if mamba_dropout is None:
            mamba_dropout = dropout

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            _rmsnorm(d_model),
        )

        # 位置编码
        pe = sinusoidal_positional_encoding(max_len, d_model, device="cpu")
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.pos_drop = nn.Dropout(dropout)

        # 堆叠并行层
        self.layers = nn.ModuleList([
            ParallelConv3BiMamba1AttnLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                conv_k=conv_k,
                mamba_dropout=mamba_dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        self.norm = _rmsnorm(d_model)

        # pooling
        self.pool = ICBHI_Pooling(d_model)

        # 这两个头：路线A用不上（因为 backbone 返回 feature），但保留兼容路线B
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.token_head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None, return_feature=False):
        """
        x: (B,T,in_dim)
        mask: (B,T) True=PAD
        return_feature=True -> (file_feature, token_features)
        """
        x = self.input_proj(x)
        T = x.shape[1]
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        file_feature = self.pool(x, mask=mask)  # (B,D)

        if return_feature:
            return file_feature, x

        file_logits = self.classifier(file_feature)
        token_logits = self.token_head(x)
        return file_logits, token_logits

# =========================
# 6) 路线A：Backbone 封装
# =========================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_HeARTokens):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model

    def forward(self, x, mask=None):
        feat, _ = self.ssa(x, mask=mask, return_feature=True)  # (B,D)
        return feat

# =========================
# 7) 工厂函数：build_backbone（路线A训练用）
# =========================
def build_backbone(in_dim=768, d_model=256, n_layers=4, nhead=8, dropout=0.3, max_len=4096,
                   conv_k=7, mamba_dropout=None, d_state=16, d_conv=4, expand=2):
    ssa = SSA_Model_HeARTokens(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
        num_classes=4,
        conv_k=conv_k,
        mamba_dropout=mamba_dropout,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    backbone = SSA_Backbone(ssa)

    params = sum(p.numel() for p in backbone.parameters())
    print("✅ Parallel Conv + (3×BiMamba || 1×Attn) Backbone Initialized.")
    print(f"   Parameters: {params:,} | Feature Dim: {backbone.final_feat_dim}")
    return backbone
