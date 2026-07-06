import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except Exception:
    Mamba = None
    HAS_MAMBA = False


# ============================================================
# Feed Forward
# ============================================================
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
            nn.Linear(
                dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                dim,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Time Mamba Block
# ============================================================
class TimeMambaBlock(nn.Module):
    """
    对每一个频率位置，沿时间轴进行 Mamba 建模。

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

        self.norm1 = nn.LayerNorm(
            dim
        )

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
            # 正式训练时应确保 HAS_MAMBA=True。
            self.sequence_model = nn.GRU(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

            self.use_mamba = False

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=dim * 2,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x_norm = self.norm1(
            x
        )

        if self.use_mamba:
            x_out = self.sequence_model(
                x_norm
            )
        else:
            x_out, _ = self.sequence_model(
                x_norm
            )

        x = residual + self.dropout(
            x_out
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# Frequency Attention Block
# ============================================================
class FrequencyAttentionBlock(nn.Module):
    """
    对每一个时间位置，沿频率轴进行多头注意力建模。

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

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=dim * 2,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x_norm = self.norm1(
            x
        )

        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False,
        )

        x = residual + self.dropout(
            attn_out
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# Time-Frequency Encoder
# ============================================================
class TimeFrequencyEncoder(nn.Module):
    """
    串联式时频编码器。

    输入：
        [B, Fp * Tp, input_dim]

    默认：
        Fp = 12
        Tp = 79
        Fp * Tp = 948

    流程：
        输入投影
            ↓
        Time-Mamba
            ↓
        Frequency-Attention
            ↓
        Attention Pooling + Max Pooling
            ↓
        [B, d_model]
    """

    def __init__(
        self,
        input_dim: int = 256,
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

        self.freq_patches = (
            freq_patches
        )

        self.time_patches = (
            time_patches
        )

        self.num_tokens = (
            freq_patches
            * time_patches
        )

        self.final_feat_dim = (
            d_model
        )

        # ====================================================
        # 输入投影
        # ====================================================
        self.input_proj = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Linear(
                input_dim,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
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
        self.time_blocks = nn.ModuleList(
            [
                TimeMambaBlock(
                    dim=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(
                    time_depth
                )
            ]
        )

        # ====================================================
        # Frequency-Attention
        # ====================================================
        self.freq_blocks = nn.ModuleList(
            [
                FrequencyAttentionBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(
                    freq_depth
                )
            ]
        )

        # ====================================================
        # Pooling
        # ====================================================
        pool_hidden = max(
            64,
            d_model // 2,
        )

        self.pool_score = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
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
                "输入必须是三维张量 [B,N,D]，"
                f"当前形状为 {tuple(x.shape)}。"
            )

        (
            batch_size,
            num_tokens,
            input_dim,
        ) = x.shape

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Token数量错误：输入为{num_tokens}，"
                f"模型要求"
                f"{self.freq_patches}×"
                f"{self.time_patches}="
                f"{self.num_tokens}。"
            )

        if input_dim != self.input_dim:
            raise ValueError(
                f"输入维度错误：输入为{input_dim}，"
                f"模型要求{self.input_dim}。"
            )

        # [B,948,input_dim]
        # → [B,948,d_model]
        x = self.input_proj(
            x
        )

        # [B,948,D]
        # → [B,12,79,D]
        x = x.reshape(
            batch_size,
            self.freq_patches,
            self.time_patches,
            self.d_model,
        )

        x = (
            x
            + self.freq_pos
            + self.time_pos
        )

        x = self.pos_dropout(
            x
        )

        # ====================================================
        # Time-Mamba
        # ====================================================
        for block in self.time_blocks:
            time_seq = x.reshape(
                batch_size
                * self.freq_patches,
                self.time_patches,
                self.d_model,
            )

            time_seq = block(
                time_seq
            )

            x = time_seq.reshape(
                batch_size,
                self.freq_patches,
                self.time_patches,
                self.d_model,
            )

        # ====================================================
        # Frequency-Attention
        # ====================================================
        for block in self.freq_blocks:
            freq_seq = x.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

            freq_seq = freq_seq.reshape(
                batch_size
                * self.time_patches,
                self.freq_patches,
                self.d_model,
            )

            freq_seq = block(
                freq_seq
            )

            freq_seq = freq_seq.reshape(
                batch_size,
                self.time_patches,
                self.freq_patches,
                self.d_model,
            )

            x = freq_seq.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

        # ====================================================
        # Pooling
        # ====================================================
        tokens = x.reshape(
            batch_size,
            self.num_tokens,
            self.d_model,
        )

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

        max_feature = torch.amax(
            tokens,
            dim=1,
        )

        feature = torch.cat(
            [
                attn_feature,
                max_feature,
            ],
            dim=-1,
        )

        feature = self.pool_fusion(
            feature
        )

        feature = self.output_norm(
            feature
        )

        return feature


# ============================================================
# SAME Padding Conv2d
# ============================================================
class SamePadConv2d(nn.Module):
    """
    支持偶数卷积核和任意步长的 SAME Padding。

    张量轴顺序：
        [B, C, T, F]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(
            kernel_size,
            int,
        ):
            kernel_size = (
                kernel_size,
                kernel_size,
            )

        if isinstance(
            stride,
            int,
        ):
            stride = (
                stride,
                stride,
            )

        if isinstance(
            dilation,
            int,
        ):
            dilation = (
                dilation,
                dilation,
            )

        self.kernel_size = tuple(
            kernel_size
        )

        self.stride = tuple(
            stride
        )

        self.dilation = tuple(
            dilation
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_t = x.shape[-2]
        input_f = x.shape[-1]

        kernel_t, kernel_f = (
            self.kernel_size
        )

        stride_t, stride_f = (
            self.stride
        )

        dilation_t, dilation_f = (
            self.dilation
        )

        output_t = math.ceil(
            input_t / stride_t
        )

        output_f = math.ceil(
            input_f / stride_f
        )

        effective_kernel_t = (
            dilation_t
            * (kernel_t - 1)
            + 1
        )

        effective_kernel_f = (
            dilation_f
            * (kernel_f - 1)
            + 1
        )

        total_pad_t = max(
            (
                output_t - 1
            )
            * stride_t
            + effective_kernel_t
            - input_t,
            0,
        )

        total_pad_f = max(
            (
                output_f - 1
            )
            * stride_f
            + effective_kernel_f
            - input_f,
            0,
        )

        pad_top = (
            total_pad_t // 2
        )

        pad_bottom = (
            total_pad_t
            - pad_top
        )

        pad_left = (
            total_pad_f // 2
        )

        pad_right = (
            total_pad_f
            - pad_left
        )

        x = F.pad(
            x,
            (
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            ),
        )

        return self.conv(
            x
        )


# ============================================================
# DTF Time-Frequency Decoupled Stem
# ============================================================
class DTFStem(nn.Module):
    """
    DTF 时频解耦 Stem。

    输入和输出：
        [B, C, T, F]

    时间分支：
        kernel=(6,3)

    频率分支：
        kernel=(3,6)

    融合方式：
        alpha * time_feature
        + (1-alpha) * freq_feature
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        time_kernel=(6, 3),
        freq_kernel=(3, 6),
    ) -> None:
        super().__init__()

        # ====================================================
        # Time Branch
        # ====================================================
        self.time_branch = nn.Sequential(
            SamePadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=time_kernel,
                stride=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),

            SamePadConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=time_kernel,
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),
        )

        # ====================================================
        # Frequency Branch
        # ====================================================
        self.freq_branch = nn.Sequential(
            SamePadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=freq_kernel,
                stride=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),

            SamePadConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=freq_kernel,
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),
        )

        # sigmoid(0)=0.5
        self.alpha_logit = nn.Parameter(
            torch.zeros(())
        )

    def get_alpha_tensor(
        self,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.alpha_logit
        )

    def get_alpha(
        self,
    ) -> float:
        return float(
            self.get_alpha_tensor()
            .detach()
            .cpu()
            .item()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        time_feature = self.time_branch(
            x
        )

        freq_feature = self.freq_branch(
            x
        )

        if (
            time_feature.shape
            != freq_feature.shape
        ):
            raise RuntimeError(
                "DTF两个分支输出尺寸不一致："
                f"time={tuple(time_feature.shape)}, "
                f"freq={tuple(freq_feature.shape)}"
            )

        alpha = self.get_alpha_tensor()

        x = (
            alpha
            * time_feature
            + (
                1.0 - alpha
            )
            * freq_feature
        )

        return x


# ============================================================
# DTF Fbank Frontend
# ============================================================
class DTFFrontend(nn.Module):
    """
    输入：
        [B,1,798,128]

    DTF Stem：
        [B,64,399,64]

    Patch Downsampling：
        [B,256,79,12]

    转换为：
        [B,948,256]

    948 = 12 × 79 个二维时频位置。
    这里不使用任何 AST 模型或 AST Token。
    """

    def __init__(
        self,
        in_channels: int = 1,
        stem_dim: int = 64,
        embed_dim: int = 256,
        freq_patches: int = 12,
        time_patches: int = 79,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.embed_dim = (
            embed_dim
        )

        self.freq_patches = (
            freq_patches
        )

        self.time_patches = (
            time_patches
        )

        self.num_tokens = (
            freq_patches
            * time_patches
        )

        self.stem = DTFStem(
            in_channels=in_channels,
            out_channels=stem_dim,
            time_kernel=(6, 3),
            freq_kernel=(3, 6),
        )

        # Stem输出为 [399,64]
        #
        # Time:
        # floor((399-5)/5)+1 = 79
        #
        # Frequency:
        # floor((64-5)/5)+1 = 12
        self.patch_downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=stem_dim,
                out_channels=embed_dim,
                kernel_size=(5, 5),
                stride=(5, 5),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                embed_dim
            ),
            nn.GELU(),
            nn.Dropout2d(
                dropout
            ),
        )

    def get_alpha(
        self,
    ) -> float:
        return self.stem.get_alpha()

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(
                "DTFFrontend输入必须是"
                "[B,C,T,F]，"
                f"当前为{tuple(x.shape)}。"
            )

        if x.shape[1] != 1:
            raise ValueError(
                "Fbank输入通道数必须为1，"
                f"当前为{x.shape[1]}。"
            )

        if tuple(
            x.shape[-2:]
        ) != (798, 128):
            raise ValueError(
                "Fbank尺寸必须为[798,128]，"
                f"当前为{tuple(x.shape[-2:])}。"
            )

        # [B,1,798,128]
        # → [B,64,399,64]
        stem_map = self.stem(
            x
        )

        # [B,64,399,64]
        # → [B,256,79,12]
        patch_map = self.patch_downsample(
            stem_map
        )

        expected_map_shape = (
            self.time_patches,
            self.freq_patches,
        )

        if tuple(
            patch_map.shape[-2:]
        ) != expected_map_shape:
            raise RuntimeError(
                "DTF Patch Map尺寸错误："
                f"当前={tuple(patch_map.shape)}，"
                f"要求空间尺寸="
                f"{expected_map_shape}。"
            )

        batch_size = (
            patch_map.shape[0]
        )

        # [B,D,T,F]
        # → [B,F,T,D]
        patch_grid = patch_map.permute(
            0,
            3,
            2,
            1,
        ).contiguous()

        # [B,F,T,D]
        # → [B,F*T,D]
        tokens = patch_grid.reshape(
            batch_size,
            self.num_tokens,
            self.embed_dim,
        )

        if return_maps:
            return (
                tokens,
                stem_map,
                patch_map,
            )

        return tokens


# ============================================================
# DTF Hybrid Model
# ============================================================
class DTFHybridModel(nn.Module):
    """
    第一阶段模型：

        Fbank
            ↓
        DTF时频解耦Stem
            ↓
        Patch Downsampling
            ↓
        Time-Mamba
            ↓
        Frequency-Attention
            ↓
        Attention + Max Pooling
            ↓
        四分类

    当前只实现Stem-TF。

    暂时不加入：
        TF-MBConv
        Window Attention
        Grid Attention
    """

    def __init__(
        self,
        num_classes: int = 4,
        stem_dim: int = 64,
        d_model: int = 256,
        freq_patches: int = 12,
        time_patches: int = 79,
        time_depth: int = 1,
        freq_depth: int = 1,
        num_heads: int = 8,
        dropout: float = 0.15,
        head_dropout: float = 0.20,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()

        self.frontend = DTFFrontend(
            in_channels=1,
            stem_dim=stem_dim,
            embed_dim=d_model,
            freq_patches=freq_patches,
            time_patches=time_patches,
            dropout=dropout,
        )

        self.encoder = TimeFrequencyEncoder(
            input_dim=d_model,
            d_model=d_model,
            freq_patches=freq_patches,
            time_patches=time_patches,
            time_depth=time_depth,
            freq_depth=freq_depth,
            num_heads=num_heads,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Dropout(
                head_dropout
            ),
            nn.Linear(
                d_model,
                num_classes,
            ),
        )

    def get_dtf_alpha(
        self,
    ) -> float:
        return self.frontend.get_alpha()

    def extract_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.frontend(
            x
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.frontend(
            x
        )

        feature = self.encoder(
            tokens
        )

        logits = self.classifier(
            feature
        )

        return logits


# ============================================================
# Shape Test
# ============================================================
if __name__ == "__main__":
    print(
        f"HAS_MAMBA = {HAS_MAMBA}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = DTFHybridModel(
        num_classes=4,
        stem_dim=64,
        d_model=256,
        freq_patches=12,
        time_patches=79,
        time_depth=1,
        freq_depth=1,
        num_heads=8,
        dropout=0.15,
        head_dropout=0.20,
    ).to(
        device
    )

    dummy_input = torch.randn(
        2,
        1,
        798,
        128,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        (
            tokens,
            stem_map,
            patch_map,
        ) = model.frontend(
            dummy_input,
            return_maps=True,
        )

        logits = model(
            dummy_input
        )

    print(
        "Device:",
        device,
    )

    print(
        "Input:",
        tuple(
            dummy_input.shape
        ),
    )

    print(
        "DTF Stem:",
        tuple(
            stem_map.shape
        ),
    )

    print(
        "Patch Map:",
        tuple(
            patch_map.shape
        ),
    )

    print(
        "Tokens:",
        tuple(
            tokens.shape
        ),
    )

    print(
        "Logits:",
        tuple(
            logits.shape
        ),
    )

    print(
        "DTF alpha:",
        model.get_dtf_alpha(),
    )

    assert tuple(
        stem_map.shape
    ) == (
        2,
        64,
        399,
        64,
    )

    assert tuple(
        patch_map.shape
    ) == (
        2,
        256,
        79,
        12,
    )

    assert tuple(
        tokens.shape
    ) == (
        2,
        948,
        256,
    )

    assert tuple(
        logits.shape
    ) == (
        2,
        4,
    )

    print(
        "DTF model shape test passed."
    )