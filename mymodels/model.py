#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Mamba
# ============================================================
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Drop Path
# ============================================================
class DropPath(nn.Module):
    def __init__(
        self,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(
                f"drop_prob必须在[0,1)内，当前为{drop_prob}。"
            )

        self.drop_prob = float(drop_prob)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.drop_prob == 0.0
            or not self.training
        ):
            return x

        keep_prob = 1.0 - self.drop_prob

        shape = (
            x.shape[0],
            *([1] * (x.ndim - 1)),
        )

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )

        random_tensor.floor_()

        return (
            x
            / keep_prob
            * random_tensor
        )


# ============================================================
# Dynamic SAME Padding Conv2d
# ============================================================
class SamePadConv2d(nn.Module):
    """
    支持偶数卷积核的动态SAME Padding。

    输入张量轴顺序：
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

        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.dilation = tuple(dilation)

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
# Squeeze-and-Excitation
# ============================================================
class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 4,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            channels // reduction_ratio,
            8,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.pool(x)
        weight = self.fc(weight)

        return x * weight


# ============================================================
# DTF Stem
# ============================================================
class DTFStem(nn.Module):
    """
    输入：
        [B,1,798,128]

    时间分支：
        kernel=(6,3)

    频率分支：
        kernel=(3,6)

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
        time_feature = self.time_branch(x)

        frequency_feature = (
            self.frequency_branch(x)
        )

        if (
            time_feature.shape
            != frequency_feature.shape
        ):
            raise RuntimeError(
                "DTF Stem两个分支尺寸不一致："
                f"time={tuple(time_feature.shape)}, "
                f"frequency={tuple(frequency_feature.shape)}"
            )

        alpha = self.get_alpha_tensor()

        return (
            alpha * time_feature
            + (
                1.0 - alpha
            ) * frequency_feature
        )


# ============================================================
# 可选TF-MBConv
# 当前实验depth=0，不会执行
# ============================================================
class TFMBConv(nn.Module):
    """
    当tf_mbconv_depth > 0时启用。

    当前B1实验：
        tf_mbconv_depth = 0
    """

    def __init__(
        self,
        channels: int,
        expand_ratio: int = 2,
        time_kernel: Tuple[int, int] = (6, 3),
        frequency_kernel: Tuple[int, int] = (3, 6),
        se_reduction: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_channels = (
            channels * expand_ratio
        )

        self.pre_norm = nn.BatchNorm2d(
            channels
        )

        self.expand_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=False,
        )

        self.expand_norm = nn.BatchNorm2d(
            hidden_channels
        )

        self.expand_activation = nn.GELU()

        self.time_depthwise = SamePadConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=time_kernel,
            stride=1,
            groups=hidden_channels,
            bias=False,
        )

        self.frequency_depthwise = SamePadConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=frequency_kernel,
            stride=1,
            groups=hidden_channels,
            bias=False,
        )

        self.beta_logit = nn.Parameter(
            torch.zeros(())
        )

        self.depthwise_norm = nn.BatchNorm2d(
            hidden_channels
        )

        self.depthwise_activation = nn.GELU()

        self.se = SqueezeExcitation(
            channels=hidden_channels,
            reduction_ratio=se_reduction,
        )

        self.project_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

        self.project_norm = nn.BatchNorm2d(
            channels
        )

        self.drop_path = DropPath(
            drop_prob=drop_path
        )

    def get_beta_tensor(
        self,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.beta_logit
        )

    def get_beta(
        self,
    ) -> float:
        return float(
            self.get_beta_tensor()
            .detach()
            .cpu()
            .item()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x = self.pre_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_activation(x)

        time_feature = self.time_depthwise(
            x
        )

        frequency_feature = (
            self.frequency_depthwise(x)
        )

        beta = self.get_beta_tensor()

        x = (
            beta * time_feature
            + (
                1.0 - beta
            ) * frequency_feature
        )

        x = self.depthwise_norm(x)
        x = self.depthwise_activation(x)

        x = self.se(x)

        x = self.project_conv(x)
        x = self.project_norm(x)

        x = self.drop_path(x)

        return residual + x


# ============================================================
# Time-Mamba Block
# ============================================================
class TimeMambaBlock(nn.Module):
    """
    每个频率位置沿时间轴执行Mamba。

    输入输出：
        [B*F,T,D]
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
            # 仅用于没有Mamba时的形状检查。
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

        self.norm2 = nn.LayerNorm(dim)

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

        x_norm = self.norm1(x)

        if self.use_mamba:
            sequence_output = (
                self.sequence_model(x_norm)
            )
        else:
            sequence_output, _ = (
                self.sequence_model(x_norm)
            )

        x = residual + self.dropout(
            sequence_output
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# Frequency-Attention Block
# ============================================================
class FrequencyAttentionBlock(nn.Module):
    """
    每个时间位置沿频率轴执行注意力。

    输入输出：
        [B*T,F,D]
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
                f"dim={dim}不能被num_heads={num_heads}整除。"
            )

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(dim)

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

        x_norm = self.norm1(x)

        attention_output, _ = (
            self.attention(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=False,
            )
        )

        x = residual + self.dropout(
            attention_output
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# Time-Mamba + Frequency-Attention Encoder
# ============================================================
class TimeFrequencyEncoder(nn.Module):
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
            freq_patches
            * time_patches
        )

        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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

        self.position_dropout = nn.Dropout(
            dropout
        )

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

        pooling_hidden = max(
            64,
            d_model // 2,
        )

        self.pooling_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                pooling_hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                pooling_hidden,
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

        self.output_norm = nn.LayerNorm(
            d_model
        )

        nn.init.trunc_normal_(
            self.frequency_position,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.time_position,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "输入必须为[B,N,D]，"
                f"当前为{tuple(x.shape)}。"
            )

        batch_size, num_tokens, input_dim = (
            x.shape
        )

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Token数量错误："
                f"{num_tokens}!={self.num_tokens}"
            )

        if input_dim != self.input_dim:
            raise ValueError(
                f"Token维度错误："
                f"{input_dim}!={self.input_dim}"
            )

        x = self.input_projection(x)

        # [B,948,D] -> [B,F,T,D]
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
        # ----------------------------------------------------
        for block in self.frequency_blocks:
            frequency_sequence = x.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

            frequency_sequence = (
                frequency_sequence.reshape(
                    batch_size
                    * self.time_patches,
                    self.freq_patches,
                    self.d_model,
                )
            )

            frequency_sequence = block(
                frequency_sequence
            )

            frequency_sequence = (
                frequency_sequence.reshape(
                    batch_size,
                    self.time_patches,
                    self.freq_patches,
                    self.d_model,
                )
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

        attention_logits = (
            self.pooling_score(tokens)
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
# DTF Frontend
# tf_mbconv_depth可设置为0
# ============================================================
class DTFFrontend(nn.Module):
    """
    当前实验设置：
        tf_mbconv_depth=0

    输入：
        [B,1,798,128]

    DTF Stem：
        [B,64,399,64]

    Patch Downsampling：
        [B,256,79,12]

    Tokens：
        [B,948,256]
    """

    def __init__(
        self,
        in_channels: int = 1,
        stem_dim: int = 64,
        embed_dim: int = 256,
        freq_patches: int = 12,
        time_patches: int = 79,
        tf_mbconv_depth: int = 0,
        tf_expand_ratio: int = 2,
        tf_se_reduction: int = 4,
        max_drop_path: float = 0.05,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if tf_mbconv_depth < 0:
            raise ValueError(
                "tf_mbconv_depth不能小于0。"
            )

        self.embed_dim = embed_dim
        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self.num_tokens = (
            freq_patches
            * time_patches
        )

        self.tf_mbconv_depth = (
            tf_mbconv_depth
        )

        self.stem = DTFStem(
            in_channels=in_channels,
            out_channels=stem_dim,
        )

        if tf_mbconv_depth == 0:
            drop_path_values = []
        else:
            drop_path_values = torch.linspace(
                0.0,
                max_drop_path,
                steps=tf_mbconv_depth,
            ).tolist()

        self.tf_mbconv_blocks = nn.ModuleList(
            [
                TFMBConv(
                    channels=stem_dim,
                    expand_ratio=tf_expand_ratio,
                    time_kernel=(6, 3),
                    frequency_kernel=(3, 6),
                    se_reduction=tf_se_reduction,
                    drop_path=drop_path_values[
                        block_index
                    ],
                )
                for block_index in range(
                    tf_mbconv_depth
                )
            ]
        )

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

    def get_all_alphas(
        self,
    ) -> Dict[str, float]:
        values = {
            "stem_alpha": (
                self.stem.get_alpha()
            )
        }

        for index, block in enumerate(
            self.tf_mbconv_blocks,
            start=1,
        ):
            values[
                f"block_{index}_beta"
            ] = block.get_beta()

        return values

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(
                "DTFFrontend输入必须为[B,C,T,F]，"
                f"当前为{tuple(x.shape)}。"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"输入通道必须为1，当前为{x.shape[1]}。"
            )

        if tuple(
            x.shape[-2:]
        ) != (
            798,
            128,
        ):
            raise ValueError(
                "Fbank尺寸必须为[798,128]，"
                f"当前为{tuple(x.shape[-2:])}。"
            )

        # [B,1,798,128]
        # -> [B,64,399,64]
        stem_map = self.stem(x)

        # depth=0时，此循环不执行
        block_map = stem_map

        for block in self.tf_mbconv_blocks:
            block_map = block(
                block_map
            )

        # [B,64,399,64]
        # -> [B,256,79,12]
        patch_map = self.patch_downsample(
            block_map
        )

        expected_map_shape = (
            self.time_patches,
            self.freq_patches,
        )

        if tuple(
            patch_map.shape[-2:]
        ) != expected_map_shape:
            raise RuntimeError(
                "Patch Map尺寸错误："
                f"当前={tuple(patch_map.shape)}，"
                f"要求={expected_map_shape}。"
            )

        batch_size = patch_map.shape[0]

        # [B,D,T,F] -> [B,F,T,D]
        patch_grid = patch_map.permute(
            0,
            3,
            2,
            1,
        ).contiguous()

        # [B,F,T,D] -> [B,F*T,D]
        tokens = patch_grid.reshape(
            batch_size,
            self.num_tokens,
            self.embed_dim,
        )

        if return_maps:
            return (
                tokens,
                stem_map,
                block_map,
                patch_map,
            )

        return tokens


# ============================================================
# 完整模型
# ============================================================
class DTFHybridModel(nn.Module):
    """
    当前B1模型：

        Fbank
          ↓
        DTF Stem
          ↓
        TF-MBConv depth=0
          ↓
        Patch Downsampling
          ↓
        Time-Mamba
          ↓
        Frequency-Attention
          ↓
        Pooling
          ↓
        四分类
    """

    def __init__(
        self,
        num_classes: int = 4,
        stem_dim: int = 64,
        d_model: int = 256,
        freq_patches: int = 12,
        time_patches: int = 79,
        tf_mbconv_depth: int = 0,
        tf_expand_ratio: int = 2,
        tf_se_reduction: int = 4,
        max_drop_path: float = 0.05,
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
            tf_mbconv_depth=tf_mbconv_depth,
            tf_expand_ratio=tf_expand_ratio,
            tf_se_reduction=tf_se_reduction,
            max_drop_path=max_drop_path,
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

    def extract_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.frontend(x)

    def get_all_alphas(
        self,
    ) -> Dict[str, float]:
        return self.frontend.get_all_alphas()

    def get_dtf_alpha(
        self,
    ) -> float:
        return self.frontend.stem.get_alpha()

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

        # 当前B1：关闭TF-MBConv
        tf_mbconv_depth=0,

        time_depth=1,
        freq_depth=1,
        num_heads=8,
        dropout=0.15,
        head_dropout=0.20,
    ).to(device)

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
            block_map,
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
        tuple(dummy_input.shape),
    )

    print(
        "DTF Stem Map:",
        tuple(stem_map.shape),
    )

    print(
        "Block Map:",
        tuple(block_map.shape),
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
        "Alphas:",
        model.get_all_alphas(),
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
        block_map.shape
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

    assert set(
        model.get_all_alphas().keys()
    ) == {
        "stem_alpha",
    }

    print(
        "B1 DTF model shape test passed."
    )