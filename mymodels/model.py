import torch
import torch.nn as nn

# 如果你本机没装 mamba_ssm，建议用 try/except 防止 IDE 报错或运行直接挂
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("⚠️ 警告：未安装 mamba_ssm（Mamba=None）。如果你模型会用到 Mamba，请先安装：pip install mamba-ssm causal-conv1d")


# =========================
# 1. 位置编码函数
# =========================
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


# =========================
# 2. 基础 BiMamba 块
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        h = self.ln1(x)
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])  # 双向
        x = x + self.drop(h_f + h_b)
        return x + self.mlp(self.ln2(x))


# =========================
# 3. SSA 层 (3 Mamba -> 1 Attention -> 3 Mamba)
# =========================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # 1) 局部特征提取（depthwise conv）
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 2) 前置 3 个 BiMamba
        self.mambas_pre = nn.ModuleList([BiMambaBlock(d_model, dropout=dropout) for _ in range(3)])

        # 3) 中间 1 个 Attention
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 4) 后置 3 个 BiMamba
        self.mambas_post = nn.ModuleList([BiMambaBlock(d_model, dropout=dropout) for _ in range(3)])

        # 5) 门控残差
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        """
        x: (B, T, D)
        mask: (B, T) True 表示 padding
        """
        res = x

        # Local Conv
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # Pre Mambas
        for blk in self.mambas_pre:
            x = blk(x)

        # Single Attention
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask)
        x = x + x_a

        # Post Mambas
        for blk in self.mambas_post:
            x = blk(x)

        # ✅ Mask-aware gated residual（避免 padding 污染）
        if mask is None:
            pooled = x.mean(dim=1, keepdim=True)  # (B,1,D)
        else:
            valid = (~mask).unsqueeze(-1).float()  # (B,T,1) 1=valid
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (x * valid).sum(dim=1, keepdim=True) / denom

        g = self.gate(pooled)  # (B,1,D)
        return res + g * x


# =========================
# 4. SSA 模型（支持返回 feature）
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
        self.num_classes = num_classes
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 位置编码 buffer（CPU 初始化，forward 时自动搬到同设备）
        pe = sinusoidal_positional_encoding(max_len, d_model, device='cpu')
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)
        self.pos_drop = nn.Dropout(dropout)

        # 堆叠 SSA 层
        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Attention Pooling（文件级 pooling）
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # （路线B兼容保留：内部 classifier/token_head，可不用）
        self.classifier = nn.Linear(d_model, num_classes)      # (B,4)
        self.token_head = nn.Linear(d_model, num_classes)      # (B,T,4)

    def forward(self, x, mask=None, return_feature=False):
        """
        x: (B, T, in_dim)
        mask: (B, T) True 表示 padding
        return_feature=True:
            file_feature: (B, d_model)
            token_feature: (B, T, d_model)
        return_feature=False:
            file_logits: (B, num_classes)
            token_logits: (B, T, num_classes)
        """
        # 1) 投影与位置编码
        x = self.input_proj(x)  # (B,T,d_model)
        B, T, D = x.shape

        # pe 自动 broadcast，且切片到 T
        x = x + self.pe[:, :T, :].to(x.device)
        x = self.pos_drop(x)

        # 2) SSA 堆叠
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        # 3) Attention pooling 得到 file_feature
        attn_scores = self.attention_net(x)  # (B, T, 1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)

        attn_w = torch.softmax(attn_scores, dim=1)   # (B, T, 1)
        file_feature = torch.sum(attn_w * x, dim=1)  # (B, d_model)

        if return_feature:
            return file_feature, x

        # 4) 输出 logits（路线B兼容）
        file_logits = self.classifier(file_feature)  # (B,4)
        token_logits = self.token_head(x)            # (B,T,4)
        return file_logits, token_logits


# =========================
# 5. Backbone Wrapper（路线A）
# =========================
class SSA_Backbone(nn.Module):
    def __init__(self, ssa_model: SSA_Model_HeARTokens):
        super().__init__()
        self.ssa = ssa_model
        self.final_feat_dim = ssa_model.d_model  # 直接等于 d_model

    def forward(self, x, mask=None):
        feat, _ = self.ssa(x, mask=mask, return_feature=True)  # (B,d_model)
        return feat


def build_backbone(in_dim=768, d_model=256, n_layers=4, nhead=8, dropout=0.3, max_len=4096):
    ssa = SSA_Model_HeARTokens(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
        num_classes=4,
    )
    backbone = SSA_Backbone(ssa)
    params = sum(p.numel() for p in backbone.parameters())
    print(f"✅ SSA Backbone Initialized. Parameters: {params:,} | final_feat_dim={backbone.final_feat_dim}")
    return backbone
