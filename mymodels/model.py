import torch
import torch.nn as nn
from mamba_ssm import Mamba


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


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
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
        # x: (B, T, D)
        h = self.ln1(x)
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        return x + self.mlp(self.ln2(x))


class SSA_Layer(nn.Module):
    """
    一个 layer 的结构变成：
    local conv
      -> BiMamba x3
      -> Self-Attention x1
      -> BiMamba x3
      -> gated residual
    """
    def __init__(self, d_model, nhead=8, dropout=0.3, mamba_blocks=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # ✅ 前 3 个 BiMamba
        self.mambas_pre = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout) for _ in range(mamba_blocks)
        ])

        # ✅ 中间 1 个 Attention
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        # ✅ 后 3 个 BiMamba
        self.mambas_post = nn.ModuleList([
            BiMambaBlock(d_model, dropout=dropout) for _ in range(mamba_blocks)
        ])

        # gated residual
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x, mask=None):
        """
        x: (B, T, D)
        mask: (B, T)  True=padding(无效位置), False=有效token
        """
        res = x

        # 0) local conv
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # 1) BiMamba x3 (pre)
        for blk in self.mambas_pre:
            x = blk(x)

        # 2) Attention x1
        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask, need_weights=False)
        x = x + x_a

        # 3) BiMamba x3 (post)
        for blk in self.mambas_post:
            x = blk(x)

        # 4) gated residual
        g = self.gate(x.mean(dim=1, keepdim=True))  # (B,1,D)
        return res + g * x



class SSA_Model_HeARTokens(nn.Module):
    """
    输入：HeAR tokens (B, T, 1024)，你现在 T≈200（也可变）
    输出：file_logit (B,), token_logits (B, T)
    """
    def __init__(self, in_dim=1024, d_model=512, n_layers=4, nhead=8, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Attention Pooling (learnable)
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        self.classifier = nn.Linear(d_model, 1)
        self.token_head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        x: (B, T, 1024)
        mask: (B, T) True=padding
        """
        x = self.input_proj(x)  # (B, T, d_model)

        B, T, D = x.shape
        pos = sinusoidal_positional_encoding(T, D, x.device).unsqueeze(0)  # (1,T,D)
        x = x + pos

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)  # (B, T, d_model)

        # Attention pooling -> file_feature
        attn_scores = self.attention_net(x)  # (B, T, 1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)

        attn_w = torch.softmax(attn_scores, dim=1)        # (B, T, 1)
        file_feature = torch.sum(attn_w * x, dim=1)       # (B, d_model)

        file_logit = self.classifier(file_feature).squeeze(-1)  # (B,)
        token_logits = self.token_head(x).squeeze(-1)           # (B, T)

        return file_logit, token_logits


def build_model(in_dim=1024, d_model=512, n_layers=4, nhead=8, dropout=0.3):
    model = SSA_Model_HeARTokens(
        in_dim=in_dim, d_model=d_model, n_layers=n_layers, nhead=nhead, dropout=dropout
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ SSA Model for HeAR tokens Initialized. Parameters: {params:,}")
    return model
