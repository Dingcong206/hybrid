import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TimeMambaBlock(nn.Module):
    """
    Time-Mamba 模块

    输入:
        x: [B * Fp, Tp, D]

    含义:
        对每一个频率 patch，沿时间轴建模。
    """

    def __init__(
        self,
        dim=768,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_mamba = True
        else:
            # 如果 mamba_ssm 没装，先用 GRU 占位，保证代码结构能跑通
            self.mamba = nn.GRU(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.use_mamba = False

        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, dropout=dropout)

    def forward(self, x):
        """
        x: [B * Fp, Tp, D]
        """

        residual = x
        x_norm = self.norm1(x)

        if self.use_mamba:
            x_out = self.mamba(x_norm)
        else:
            x_out, _ = self.mamba(x_norm)

        x = residual + self.dropout(x_out)

        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class FrequencyAttentionBlock(nn.Module):
    """
    Frequency-Attention 模块

    输入:
        x: [B * Tp, Fp, D]

    含义:
        对每一个时间 patch，沿频率轴建模。
    """

    def __init__(
        self,
        dim=768,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, dropout=dropout)

    def forward(self, x):
        """
        x: [B * Tp, Fp, D]
        """

        residual = x
        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False
        )

        x = residual + self.dropout(attn_out)

        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class TimeFrequencyEncoder(nn.Module):
    """
    Time-Mamba + Frequency-Attention 特征提取模型

    输入:
        x: [B, N, D]

    例如:
        x: [B, 948, 768]

    其中:
        948 = 12 * 79
        Fp = 12
        Tp = 79

    输出:
        feature: [B, D]
    """

    def __init__(
        self,
        token_dim=768,
        freq_patches=12,
        time_patches=79,
        time_depth=2,
        freq_depth=2,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.token_dim = token_dim
        self.freq_patches = freq_patches
        self.time_patches = time_patches
        self.num_tokens = freq_patches * time_patches

        self.input_norm = nn.LayerNorm(token_dim)

        self.time_blocks = nn.ModuleList([
            TimeMambaBlock(
                dim=token_dim,
                dropout=dropout
            )
            for _ in range(time_depth)
        ])

        self.freq_blocks = nn.ModuleList([
            FrequencyAttentionBlock(
                dim=token_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(freq_depth)
        ])

        self.fusion = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        """
        x: [B, N, D]

        return:
            feature: [B, D]
        """

        B, N, D = x.shape

        if N != self.num_tokens:
            raise ValueError(
                f"Token 数量不匹配: 输入 N={N}, "
                f"但模型需要 {self.freq_patches} * {self.time_patches} = {self.num_tokens}"
            )

        if D != self.token_dim:
            raise ValueError(
                f"Token 维度不匹配: 输入 D={D}, 但模型 token_dim={self.token_dim}"
            )

        x = self.input_norm(x)

        # [B, N, D] -> [B, Fp, Tp, D]
        x = x.reshape(
            B,
            self.freq_patches,
            self.time_patches,
            self.token_dim
        )

        # =====================================================
        # 1. Time-Mamba 分支
        # =====================================================
        time_x = x

        for block in self.time_blocks:
            # [B, Fp, Tp, D] -> [B * Fp, Tp, D]
            time_seq = time_x.reshape(
                B * self.freq_patches,
                self.time_patches,
                self.token_dim
            )

            time_seq = block(time_seq)

            # [B * Fp, Tp, D] -> [B, Fp, Tp, D]
            time_x = time_seq.reshape(
                B,
                self.freq_patches,
                self.time_patches,
                self.token_dim
            )

        # =====================================================
        # 2. Frequency-Attention 分支
        # =====================================================
        freq_x = x

        for block in self.freq_blocks:
            # [B, Fp, Tp, D] -> [B, Tp, Fp, D]
            freq_seq = freq_x.permute(0, 2, 1, 3).contiguous()

            # [B, Tp, Fp, D] -> [B * Tp, Fp, D]
            freq_seq = freq_seq.reshape(
                B * self.time_patches,
                self.freq_patches,
                self.token_dim
            )

            freq_seq = block(freq_seq)

            # [B * Tp, Fp, D] -> [B, Tp, Fp, D]
            freq_seq = freq_seq.reshape(
                B,
                self.time_patches,
                self.freq_patches,
                self.token_dim
            )

            # [B, Tp, Fp, D] -> [B, Fp, Tp, D]
            freq_x = freq_seq.permute(0, 2, 1, 3).contiguous()

        # =====================================================
        # 3. 池化得到两个分支的全局特征
        # =====================================================
        time_feature = time_x.mean(dim=(1, 2))   # [B, D]
        freq_feature = freq_x.mean(dim=(1, 2))   # [B, D]

        # =====================================================
        # 4. 融合时频特征
        # =====================================================
        feature = torch.cat([time_feature, freq_feature], dim=-1)  # [B, 2D]
        feature = self.fusion(feature)                             # [B, D]
        feature = self.output_norm(feature)

        return feature