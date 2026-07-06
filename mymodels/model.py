import math
from typing import Optional, Tuple

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
# Feed Forward Network
# ============================================================
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
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


# ============================================================
# Time-Mamba Block
# ============================================================
class TimeMambaBlock(nn.Module):
    """
    对每一个频率位置，沿时间维度执行 Mamba。

    输入和输出：
        [B * F, T, D]
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
            # 仅用于没有安装 mamba_ssm 时的形状测试。
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
        residual = x
        x_norm = self.norm1(x)

        if self.use_mamba:
            sequence_output = self.sequence_model(x_norm)
        else:
            sequence_output, _ = self.sequence_model(x_norm)

        x = residual + self.dropout(sequence_output)
        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================
# Frequency-Attention Block
# ============================================================
class FrequencyAttentionBlock(nn.Module):
    """
    对每一个时间位置，沿频率维度执行多头注意力。

    输入和输出：
        [B * T, F, D]
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
                f"dim={dim} 必须能够被 num_heads={num_heads} 整除。"
            )

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
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
        residual = x
        x_norm = self.norm1(x)

        attention_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False,
        )

        x = residual + self.dropout(attention_output)
        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================
# Time-Mamba + Frequency-Attention Encoder
# ============================================================
class TimeFrequencyEncoder(nn.Module):
    """
    输入：
        [B, Fp * Tp, input_dim]

    默认：
        Fp = 12
        Tp = 79
        Token 数量 = 948

    流程：
        Input Projection
            ↓
        Time-Mamba
            ↓
        Frequency-Attention
            ↓
        Attention Pooling + Max Pooling

    输出：
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

        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self.num_tokens = (
            self.freq_patches
            * self.time_patches
        )

        self.final_feat_dim = d_model

        # ----------------------------------------------------
        # 输入投影
        # ----------------------------------------------------
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # 二维时频位置编码
        # ----------------------------------------------------
        self.frequency_position = nn.Parameter(
            torch.zeros(
                1,
                freq_patches,
                1,
                d_model,
            )
        )

        self.time_position = nn.Parameter(
            torch.zeros(
                1,
                1,
                time_patches,
                d_model,
            )
        )

        self.position_dropout = nn.Dropout(dropout)

        # ----------------------------------------------------
        # Time-Mamba
        # ----------------------------------------------------
        self.time_blocks = nn.ModuleList(
            [
                TimeMambaBlock(
                    dim=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(time_depth)
            ]
        )

        # ----------------------------------------------------
        # Frequency-Attention
        # ----------------------------------------------------
        self.frequency_blocks = nn.ModuleList(
            [
                FrequencyAttentionBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(freq_depth)
            ]
        )

        # ----------------------------------------------------
        # Attention + Max Pooling
        # ----------------------------------------------------
        pool_hidden_dim = max(
            64,
            d_model // 2,
        )

        self.pooling_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                pool_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                pool_hidden_dim,
                1,
            ),
        )

        self.pooling_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(
                d_model * 2,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(
            self.frequency_position,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.time_position,
            std=0.02,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "TimeFrequencyEncoder 输入必须是 [B,N,D]，"
                f"当前为 {tuple(x.shape)}。"
            )

        batch_size, num_tokens, input_dim = x.shape

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Token 数量错误：输入={num_tokens}，"
                f"模型要求={self.num_tokens}。"
            )

        if input_dim != self.input_dim:
            raise ValueError(
                f"Token 维度错误：输入={input_dim}，"
                f"模型要求={self.input_dim}。"
            )

        # [B,948,input_dim]
        # → [B,948,d_model]
        x = self.input_projection(x)

        # [B,948,D]
        # → [B,F,T,D]
        x = x.reshape(
            batch_size,
            self.freq_patches,
            self.time_patches,
            self.d_model,
        )

        x = (
            x
            + self.frequency_position
            + self.time_position
        )

        x = self.position_dropout(x)

        # ----------------------------------------------------
        # Time-Mamba
        # 每一个频率位置单独沿时间建模
        # ----------------------------------------------------
        for block in self.time_blocks:
            time_sequence = x.reshape(
                batch_size
                * self.freq_patches,
                self.time_patches,
                self.d_model,
            )

            time_sequence = block(
                time_sequence
            )

            x = time_sequence.reshape(
                batch_size,
                self.freq_patches,
                self.time_patches,
                self.d_model,
            )

        # ----------------------------------------------------
        # Frequency-Attention
        # 每一个时间位置单独沿频率建模
        # ----------------------------------------------------
        for block in self.frequency_blocks:
            frequency_sequence = x.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

            frequency_sequence = frequency_sequence.reshape(
                batch_size
                * self.time_patches,
                self.freq_patches,
                self.d_model,
            )

            frequency_sequence = block(
                frequency_sequence
            )

            frequency_sequence = frequency_sequence.reshape(
                batch_size,
                self.time_patches,
                self.freq_patches,
                self.d_model,
            )

            x = frequency_sequence.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------
        tokens = x.reshape(
            batch_size,
            self.num_tokens,
            self.d_model,
        )

        attention_logits = self.pooling_score(
            tokens
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        attention_feature = torch.sum(
            tokens * attention_weights,
            dim=1,
        )

        max_feature = torch.amax(
            tokens,
            dim=1,
        )

        feature = torch.cat(
            [
                attention_feature,
                max_feature,
            ],
            dim=-1,
        )

        feature = self.pooling_fusion(
            feature
        )

        feature = self.output_norm(
            feature
        )

        return feature


# ============================================================
# Dynamic SAME Padding Conv2d
# ============================================================
class SamePadConv2d(nn.Module):
    """
    支持偶数卷积核的动态 SAME Padding。

    输入张量顺序：
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

        if isinstance(kernel_size, int):
            kernel_size = (
                kernel_size,
                kernel_size,
            )

        if isinstance(stride, int):
            stride = (
                stride,
                stride,
            )

        if isinstance(dilation, int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_time = x.shape[-2]
        input_frequency = x.shape[-1]

        kernel_time, kernel_frequency = (
            self.kernel_size
        )

        stride_time, stride_frequency = (
            self.stride
        )

        dilation_time, dilation_frequency = (
            self.dilation
        )

        output_time = math.ceil(
            input_time / stride_time
        )

        output_frequency = math.ceil(
            input_frequency / stride_frequency
        )

        effective_kernel_time = (
            dilation_time
            * (kernel_time - 1)
            + 1
        )

        effective_kernel_frequency = (
            dilation_frequency
            * (kernel_frequency - 1)
            + 1
        )

        total_padding_time = max(
            (
                output_time - 1
            )
            * stride_time
            + effective_kernel_time
            - input_time,
            0,
        )

        total_padding_frequency = max(
            (
                output_frequency - 1
            )
            * stride_frequency
            + effective_kernel_frequency
            - input_frequency,
            0,
        )

        padding_top = (
            total_padding_time // 2
        )

        padding_bottom = (
            total_padding_time
            - padding_top
        )

        padding_left = (
            total_padding_frequency // 2
        )

        padding_right = (
            total_padding_frequency
            - padding_left
        )

        x = F.pad(
            x,
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
            ),
        )

        return self.conv(x)


# ============================================================
# B0: No-TF Parameter-Matched Stem
# ============================================================
class NoTFStem(nn.Module):
    """
    单分支参数匹配基线。

    DTF 每一层两个卷积核的面积：
        6×3 + 3×6 = 36

    No-TF 单分支卷积核：
        6×6 = 36

    输入：
        [B,1,798,128]

    输出：
        [B,64,399,64]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
    ) -> None:
        super().__init__()

        self.branch = nn.Sequential(
            SamePadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(6, 6),
                stride=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            SamePadConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(6, 6),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


# ============================================================
# B1: DTF Time-Frequency Decoupled Stem
# ============================================================
class DTFStem(nn.Module):
    """
    DTF 时频解耦 Stem。

    时间分支：
        kernel=(6,3)

    频率分支：
        kernel=(3,6)

    融合：
        alpha * time_feature
        + (1-alpha) * frequency_feature

    输入：
        [B,1,798,128]

    输出：
        [B,64,399,64]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        time_kernel: Tuple[int, int] = (6, 3),
        frequency_kernel: Tuple[int, int] = (3, 6),
    ) -> None:
        super().__init__()

        # ----------------------------------------------------
        # Time Branch
        # ----------------------------------------------------
        self.time_branch = nn.Sequential(
            SamePadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=time_kernel,
                stride=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            SamePadConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=time_kernel,
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Frequency Branch
        # ----------------------------------------------------
        self.frequency_branch = nn.Sequential(
            SamePadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=frequency_kernel,
                stride=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            SamePadConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=frequency_kernel,
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # sigmoid(0) = 0.5
        self.alpha_logit = nn.Parameter(
            torch.zeros(())
        )

    def get_alpha_tensor(self) -> torch.Tensor:
        return torch.sigmoid(
            self.alpha_logit
        )

    def get_alpha(self) -> float:
        return float(
            self.get_alpha_tensor()
            .detach()
            .cpu()
            .item()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_feature = self.time_branch(x)

        frequency_feature = self.frequency_branch(
            x
        )

        if (
            time_feature.shape
            != frequency_feature.shape
        ):
            raise RuntimeError(
                "DTF 两个分支输出尺寸不一致："
                f"time={tuple(time_feature.shape)}, "
                f"frequency={tuple(frequency_feature.shape)}"
            )

        alpha = self.get_alpha_tensor()

        output = (
            alpha * time_feature
            + (
                1.0 - alpha
            ) * frequency_feature
        )

        return output


# ============================================================
# Fbank Frontend
# ============================================================
class FbankFrontend(nn.Module):
    """
    frontend_type="no_tf"：
        使用单分支 6×6 Stem。

    frontend_type="dtf"：
        使用 DTF 双分支 Stem。

    输入：
        [B,1,798,128]

    Stem：
        [B,64,399,64]

    Patch Map：
        [B,256,79,12]

    Tokens：
        [B,948,256]
    """

    def __init__(
        self,
        frontend_type: str = "no_tf",
        in_channels: int = 1,
        stem_dim: int = 64,
        embed_dim: int = 256,
        freq_patches: int = 12,
        time_patches: int = 79,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        frontend_type = (
            frontend_type
            .strip()
            .lower()
        )

        if frontend_type not in {
            "no_tf",
            "dtf",
        }:
            raise ValueError(
                "frontend_type 必须为 "
                "'no_tf' 或 'dtf'，"
                f"当前为 {frontend_type!r}。"
            )

        self.frontend_type = frontend_type

        self.embed_dim = embed_dim
        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self.num_tokens = (
            freq_patches
            * time_patches
        )

        if frontend_type == "no_tf":
            self.stem = NoTFStem(
                in_channels=in_channels,
                out_channels=stem_dim,
            )
        else:
            self.stem = DTFStem(
                in_channels=in_channels,
                out_channels=stem_dim,
            )

        # Stem 输出：[399,64]
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
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def get_alpha(self) -> Optional[float]:
        if isinstance(
            self.stem,
            DTFStem,
        ):
            return self.stem.get_alpha()

        return None

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(
                "FbankFrontend 输入必须为 [B,C,T,F]，"
                f"当前为 {tuple(x.shape)}。"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"Fbank 通道数必须为1，当前为 {x.shape[1]}。"
            )

        if tuple(x.shape[-2:]) != (
            798,
            128,
        ):
            raise ValueError(
                "Fbank 尺寸必须为 [798,128]，"
                f"当前为 {tuple(x.shape[-2:])}。"
            )

        # [B,1,798,128]
        # → [B,64,399,64]
        stem_map = self.stem(x)

        # [B,64,399,64]
        # → [B,256,79,12]
        patch_map = self.patch_downsample(
            stem_map
        )

        expected_shape = (
            self.time_patches,
            self.freq_patches,
        )

        if tuple(
            patch_map.shape[-2:]
        ) != expected_shape:
            raise RuntimeError(
                "Patch Map 尺寸错误："
                f"当前={tuple(patch_map.shape)}，"
                f"要求空间尺寸={expected_shape}。"
            )

        batch_size = patch_map.shape[0]

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
# Full Hybrid Model
# ============================================================
class FbankHybridModel(nn.Module):
    """
    完整结构：

        Fbank
          ↓
        No-TF Stem 或 DTF Stem
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
    """

    def __init__(
        self,
        frontend_type: str = "no_tf",
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

        self.frontend_type = (
            frontend_type
            .strip()
            .lower()
        )

        self.frontend = FbankFrontend(
            frontend_type=self.frontend_type,
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
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(
                d_model,
                num_classes,
            ),
        )

    def get_frontend_alpha(
        self,
    ) -> Optional[float]:
        return self.frontend.get_alpha()

    def get_dtf_alpha(
        self,
    ) -> Optional[float]:
        return self.get_frontend_alpha()

    def extract_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.frontend(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.frontend(x)

        feature = self.encoder(
            tokens
        )

        logits = self.classifier(
            feature
        )

        return logits


# ============================================================
# Compatibility Classes
# ============================================================
class DTFHybridModel(FbankHybridModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        kwargs["frontend_type"] = "dtf"
        super().__init__(
            *args,
            **kwargs,
        )


class NoTFHybridModel(FbankHybridModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        kwargs["frontend_type"] = "no_tf"
        super().__init__(
            *args,
            **kwargs,
        )


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

    dummy_input = torch.randn(
        2,
        1,
        798,
        128,
        device=device,
    )

    for frontend_type in (
        "no_tf",
        "dtf",
    ):
        model = FbankHybridModel(
            frontend_type=frontend_type,
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
        ).to(device)

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

        print()
        print(
            "Frontend:",
            frontend_type,
        )

        print(
            "Input:",
            tuple(dummy_input.shape),
        )

        print(
            "Stem Map:",
            tuple(stem_map.shape),
        )

        print(
            "Patch Map:",
            tuple(patch_map.shape),
        )

        print(
            "Tokens:",
            tuple(tokens.shape),
        )

        print(
            "Logits:",
            tuple(logits.shape),
        )

        print(
            "Alpha:",
            model.get_frontend_alpha(),
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

    print()
    print(
        "All shape tests passed."
    )