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
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        if Mamba is None:
            raise RuntimeError(
                "当前环境没有安装 mamba_ssm，无法使用 BiMambaBlock。\n"
                "请安装：pip install mamba-ssm causal-conv1d"
            )

        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
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
        h = self.ln1(x)
        # 双向 Mamba 合并
        h_f = self.fwd(h)
        h_b = torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h_f + h_b)
        return x + self.mlp(self.ln2(x))


# =========================
# 3. SSA 层 (单层定义)
# =========================
class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()

        # 1) 局部特征提取
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 2) 前置 4 个 BiMamba
        self.mambas_pre = nn.ModuleList([BiMambaBlock(d_model, dropout=dropout) for _ in range(4)])

        # 3) 中间 2 个 Attention 层
        self.attn1_ln = nn.LayerNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.attn2_ln = nn.LayerNorm(d_model)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # 4) 后置 4 个 BiMamba
        self.mambas_post = nn.ModuleList([BiMambaBlock(d_model, dropout=dropout) for _ in range(4)])

        # 5) 门控残差
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        res = x

        # Local Conv
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # Pre Mambas
        for blk in self.mambas_pre:
            x = blk(x)

        # Dual Attention
        x_n1 = self.attn1_ln(x)
        x_a1, _ = self.attn1(x_n1, x_n1, x_n1, key_padding_mask=mask)
        x = x + x_a1

        x_n2 = self.attn2_ln(x)
        x_a2, _ = self.attn2(x_n2, x_n2, x_n2, key_padding_mask=mask)
        x = x + x_a2

        # Post Mambas
        for blk in self.mambas_post:
            x = blk(x)

        # Gated Residual
        g = self.gate(x.mean(dim=1, keepdim=True))
        return res + g * x


# =========================
# 4. 完整的 SSA 模型（方案B：直接输出 4类 logits）
# =========================
class SSA_Model_HeARTokens(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        d_model=512,
        n_layers=6,
        nhead=8,
        dropout=0.3,
        max_len=2000,
        num_classes=4,   # ✅ 关键：4-class
    ):
        super().__init__()
        self.num_classes = num_classes

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 位置编码 buffer
        pe = sinusoidal_positional_encoding(max_len, d_model, device='cpu')
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        self.pos_drop = nn.Dropout(dropout)

        # 堆叠 SSA 层
        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Attention Pooling
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # ✅ 关键：输出 4 类
        self.classifier = nn.Linear(d_model, num_classes)      # (B,4)
        self.token_head = nn.Linear(d_model, num_classes)      # (B,T,4) 可选

    def forward(self, x, mask=None):
        """
        x: (B, T, in_dim)
        mask: (B, T) True 表示 padding
        return:
          file_logits: (B, num_classes)
          token_logits: (B, T, num_classes)
        """
        # 1) 投影与位置编码
        x = self.input_proj(x)
        B, T, D = x.shape
        x = x + self.pe[:, :T, :]
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

        # 4) 输出 logits（✅不 squeeze）
        file_logits = self.classifier(file_feature)  # (B,4)
        token_logits = self.token_head(x)            # (B,T,4)

        return file_logits, token_logits


# =========================
# 5. 构建函数：直接返回 4-class 模型（不需要 wrapper）
# =========================
def build_model(in_dim=1024, d_model=512, n_layers=4, nhead=8, dropout=0.3, num_classes=4):
    model = SSA_Model_HeARTokens(
        in_dim=in_dim,
        d_model=d_model,
        n_layers=n_layers,
        nhead=nhead,
        dropout=dropout,
        num_classes=num_classes
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SSA 4-Class Model Initialized. Parameters: {params:,}")
    return model
