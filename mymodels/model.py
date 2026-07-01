import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or dim * 2

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeMambaBlock(nn.Module):
    """
    对每一个频率 Patch，沿时间轴进行 Mamba 建模。

    输入和输出：
        [B * Fp, Tp, D]
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        if HAS_MAMBA:
            self.sequence_model = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba = True

        else:
            # 仅用于没有安装 mamba_ssm 时进行形状检查。
            # 正式训练时 train.py 会阻止使用该分支。
            self.sequence_model = nn.GRU(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.use_mamba = False

        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=dim * 2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mamba + Residual
        residual = x
        x_norm = self.norm1(x)

        if self.use_mamba:
            x_out = self.sequence_model(x_norm)
        else:
            x_out, _ = self.sequence_model(x_norm)

        x = residual + self.dropout(x_out)

        # FFN + Residual
        x = x + self.ffn(
            self.norm2(x)
        )

        return x


class FrequencyAttentionBlock(nn.Module):
    """
    对每一个时间 Patch，沿频率轴进行多头注意力建模。

    输入和输出：
        [B * Tp, Fp, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} 必须能够被 "
                f"num_heads={num_heads} 整除。"
            )

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=dim * 2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency Attention + Residual
        residual = x
        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False,
        )

        x = residual + self.dropout(
            attn_out
        )

        # FFN + Residual
        x = x + self.ffn(
            self.norm2(x)
        )

        return x


class TimeFrequencyEncoder(nn.Module):
    """
    串联式时频编码器：

        AST Tokens
        [B, 948, 768]
                ↓
        Input Projection
        [B, 948, 256]
                ↓
        Time-Mamba
                ↓
        Frequency-Attention
                ↓
        Attention Pooling + Max Pooling
                ↓
        [B, 256]

    默认：

        948 = 12 × 79

        Fp = 12
        Tp = 79
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,
        freq_patches: int = 12,
        time_patches: int = 79,
        time_depth: int = 1,
        freq_depth: int = 1,
        num_heads: int = 8,
        dropout: float = 0.15,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self.num_tokens = (
            freq_patches * time_patches
        )

        # 外部分类器读取该属性
        self.final_feat_dim = d_model

        # ====================================================
        # 输入投影：768 → 256
        # ====================================================
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ====================================================
        # 二维位置编码
        # ====================================================
        self.freq_pos = nn.Parameter(
            torch.zeros(
                1,
                freq_patches,
                1,
                d_model,
            )
        )

        self.time_pos = nn.Parameter(
            torch.zeros(
                1,
                1,
                time_patches,
                d_model,
            )
        )

        self.pos_dropout = nn.Dropout(
            dropout
        )

        # ====================================================
        # Time-Mamba
        # ====================================================
        self.time_blocks = nn.ModuleList([
            TimeMambaBlock(
                dim=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(time_depth)
        ])

        # ====================================================
        # Frequency-Attention
        # ====================================================
        self.freq_blocks = nn.ModuleList([
            FrequencyAttentionBlock(
                dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(freq_depth)
        ])

        # ====================================================
        # Attention Pooling
        # ====================================================
        pool_hidden = max(
            64,
            d_model // 2,
        )

        self.pool_score = nn.Sequential(
            nn.LayerNorm(d_model),

            nn.Linear(
                d_model,
                pool_hidden,
            ),

            nn.Tanh(),

            nn.Linear(
                pool_hidden,
                1,
            ),
        )

        # Attention Pooling 与 Max Pooling 融合
        self.pool_fusion = nn.Sequential(
            nn.LayerNorm(
                d_model * 2
            ),

            nn.Linear(
                d_model * 2,
                d_model,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )

        nn.init.trunc_normal_(
            self.freq_pos,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.time_pos,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "输入必须是三维张量 [B, N, D]，"
                f"当前形状为 {tuple(x.shape)}。"
            )

        (
            batch_size,
            num_tokens,
            input_dim,
        ) = x.shape

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Token 数量错误：输入为 {num_tokens}，"
                f"模型要求 "
                f"{self.freq_patches} × "
                f"{self.time_patches} = "
                f"{self.num_tokens}。"
            )

        if input_dim != self.input_dim:
            raise ValueError(
                f"输入维度错误：输入为 {input_dim}，"
                f"模型要求 {self.input_dim}。"
            )

        # ====================================================
        # 输入投影
        #
        # [B, 948, 768]
        #       ↓
        # [B, 948, 256]
        # ====================================================
        x = self.input_proj(x)

        # ====================================================
        # 恢复二维时频结构
        #
        # [B, 948, 256]
        #       ↓
        # [B, 12, 79, 256]
        # ====================================================
        x = x.reshape(
            batch_size,
            self.freq_patches,
            self.time_patches,
            self.d_model,
        )

        # 加入时间和频率位置编码
        x = (
            x
            + self.freq_pos
            + self.time_pos
        )

        x = self.pos_dropout(x)

        # ====================================================
        # Time-Mamba
        #
        # 固定频率位置，沿时间轴建模
        # ====================================================
        for block in self.time_blocks:

            # [B, Fp, Tp, D]
            #       ↓
            # [B × Fp, Tp, D]
            time_seq = x.reshape(
                batch_size
                * self.freq_patches,

                self.time_patches,

                self.d_model,
            )

            time_seq = block(
                time_seq
            )

            # [B × Fp, Tp, D]
            #       ↓
            # [B, Fp, Tp, D]
            x = time_seq.reshape(
                batch_size,
                self.freq_patches,
                self.time_patches,
                self.d_model,
            )

        # ====================================================
        # Frequency-Attention
        #
        # 接收 Time-Mamba 的输出
        # ====================================================
        for block in self.freq_blocks:

            # [B, Fp, Tp, D]
            #       ↓
            # [B, Tp, Fp, D]
            freq_seq = x.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

            # [B, Tp, Fp, D]
            #       ↓
            # [B × Tp, Fp, D]
            freq_seq = freq_seq.reshape(
                batch_size
                * self.time_patches,

                self.freq_patches,

                self.d_model,
            )

            freq_seq = block(
                freq_seq
            )

            # [B × Tp, Fp, D]
            #       ↓
            # [B, Tp, Fp, D]
            freq_seq = freq_seq.reshape(
                batch_size,
                self.time_patches,
                self.freq_patches,
                self.d_model,
            )

            # [B, Tp, Fp, D]
            #       ↓
            # [B, Fp, Tp, D]
            x = freq_seq.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

        # ====================================================
        # Pooling
        #
        # [B, Fp, Tp, D]
        #       ↓
        # [B, 948, D]
        # ====================================================
        tokens = x.reshape(
            batch_size,
            self.num_tokens,
            self.d_model,
        )

        # Attention Pooling
        attn_logits = self.pool_score(
            tokens
        )

        attn_weights = torch.softmax(
            attn_logits,
            dim=1,
        )

        attn_feature = torch.sum(
            tokens * attn_weights,
            dim=1,
        )

        # Max Pooling
        max_feature = torch.amax(
            tokens,
            dim=1,
        )

        # [B, 256] + [B, 256]
        #       ↓
        # [B, 512]
        feature = torch.cat(
            [
                attn_feature,
                max_feature,
            ],
            dim=-1,
        )

        # [B, 512] → [B, 256]
        feature = self.pool_fusion(
            feature
        )

        feature = self.output_norm(
            feature
        )

        return feature


if __name__ == "__main__":

    print(
        f"HAS_MAMBA = {HAS_MAMBA}"
    )

    model = TimeFrequencyEncoder(
        input_dim=768,
        d_model=256,
        freq_patches=12,
        time_patches=79,
        time_depth=1,
        freq_depth=1,
        num_heads=8,
        dropout=0.15,
    )

    dummy_input = torch.randn(
        1,
        948,
        768,
    )

    with torch.no_grad():
        dummy_output = model(
            dummy_input
        )

    print(
        "Input shape:",
        tuple(dummy_input.shape),
    )

    print(
        "Output shape:",
        tuple(dummy_output.shape),
    )