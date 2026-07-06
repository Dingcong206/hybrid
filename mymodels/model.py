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

        if hidden_dim is None:
            hidden_dim = dim * 2

        self.network = nn.Sequential(
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
        return self.network(x)


# ============================================================
# Dynamic SAME Padding Conv2d
# ============================================================
class SamePadConv2d(nn.Module):
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
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

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

        kernel_time, kernel_frequency = self.kernel_size
        stride_time, stride_frequency = self.stride
        dilation_time, dilation_frequency = self.dilation

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
            (output_time - 1)
            * stride_time
            + effective_kernel_time
            - input_time,
            0,
        )

        total_padding_frequency = max(
            (output_frequency - 1)
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
# DTF Stem
# ============================================================
class DTFStem(nn.Module):
    """
    输入：
        [B, 1, 798, 128]

    输出：
        [B, 64, 399, 64]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        time_kernel: Tuple[int, int] = (
            6,
            3,
        ),
        frequency_kernel: Tuple[int, int] = (
            3,
            6,
        ),
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
                "DTF两个分支输出尺寸不一致："
                f"time={tuple(time_feature.shape)}, "
                f"frequency={tuple(frequency_feature.shape)}"
            )

        alpha = self.get_alpha_tensor()

        return (
            alpha * time_feature
            + (1.0 - alpha)
            * frequency_feature
        )


# ============================================================
# Residual Convolution Block
# ============================================================
class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.block(x)


# ============================================================
# Progressive Downsampling
# ============================================================
class ProgressiveDownsample(nn.Module):
    """
    输入：
        [B,64,399,64]

    Stage1：
        [B,96,200,32]

    Stage2：
        [B,160,100,16]

    Patch：
        [B,256,100,16]
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                96,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(96),
            nn.GELU(),

            ResidualConvBlock(
                channels=96,
                dropout=dropout * 0.25,
            ),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(
                96,
                160,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(160),
            nn.GELU(),

            ResidualConvBlock(
                channels=160,
                dropout=dropout * 0.50,
            ),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(
                160,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            ResidualConvBlock(
                channels=out_channels,
                dropout=dropout,
            ),

            nn.Dropout2d(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stage_maps: bool = False,
    ):
        stage1_map = self.stage1(x)
        stage2_map = self.stage2(stage1_map)
        patch_map = self.stage3(stage2_map)

        if return_stage_maps:
            return (
                patch_map,
                stage1_map,
                stage2_map,
            )

        return patch_map


# ============================================================
# Time Mamba Block
# ============================================================
class TimeMambaBlock(nn.Module):
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
            self.sequence_model = nn.GRU(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

            self.use_mamba = False

        self.sequence_dropout = nn.Dropout(
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
        normalized = self.norm1(x)

        if self.use_mamba:
            sequence_output = (
                self.sequence_model(normalized)
            )
        else:
            sequence_output, _ = (
                self.sequence_model(normalized)
            )

        x = (
            x
            + self.sequence_dropout(
                sequence_output
            )
        )

        x = (
            x
            + self.ffn(
                self.norm2(x)
            )
        )

        return x


# ============================================================
# Frequency Attention Block
# ============================================================
class FrequencyAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim}不能被"
                f"num_heads={num_heads}整除。"
            )

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(
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
        normalized = self.norm1(x)

        attention_output, _ = (
            self.attention(
                query=normalized,
                key=normalized,
                value=normalized,
                need_weights=False,
            )
        )

        x = (
            x
            + self.attention_dropout(
                attention_output
            )
        )

        x = (
            x
            + self.ffn(
                self.norm2(x)
            )
        )

        return x


# ============================================================
# Time Frequency Encoder
# ============================================================
class TimeFrequencyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 256,
        freq_patches: int = 16,
        time_patches: int = 100,
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
            nn.LayerNorm(
                d_model * 2
            ),
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
                "Encoder输入必须为[B,N,D]，"
                f"当前为{tuple(x.shape)}。"
            )

        batch_size, num_tokens, input_dim = (
            x.shape
        )

        if num_tokens != self.num_tokens:
            raise ValueError(
                "Token数量错误："
                f"当前={num_tokens}，"
                f"要求={self.num_tokens}。"
            )

        if input_dim != self.input_dim:
            raise ValueError(
                "Token维度错误："
                f"当前={input_dim}，"
                f"要求={self.input_dim}。"
            )

        x = self.input_projection(x)

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

        # Time-Mamba
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

        # Frequency-Attention
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
# ============================================================
class DTFFrontend(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        stem_dim: int = 64,
        embed_dim: int = 256,
        freq_patches: int = 16,
        time_patches: int = 100,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self.num_tokens = (
            freq_patches
            * time_patches
        )

        self.stem = DTFStem(
            in_channels=in_channels,
            out_channels=stem_dim,
        )

        self.progressive_downsample = (
            ProgressiveDownsample(
                in_channels=stem_dim,
                out_channels=embed_dim,
                dropout=dropout,
            )
        )

    def get_alpha(
        self,
    ) -> float:
        return self.stem.get_alpha()

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
        return_stage_maps: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(
                "Frontend输入必须为[B,C,T,F]，"
                f"当前为{tuple(x.shape)}。"
            )

        if x.shape[1] != 1:
            raise ValueError(
                "输入通道数必须为1，"
                f"当前为{x.shape[1]}。"
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

        stem_map = self.stem(x)

        if return_stage_maps:
            (
                patch_map,
                stage1_map,
                stage2_map,
            ) = self.progressive_downsample(
                stem_map,
                return_stage_maps=True,
            )
        else:
            patch_map = (
                self.progressive_downsample(
                    stem_map
                )
            )

            stage1_map = None
            stage2_map = None

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

        patch_grid = patch_map.permute(
            0,
            3,
            2,
            1,
        ).contiguous()

        tokens = patch_grid.reshape(
            batch_size,
            self.num_tokens,
            self.embed_dim,
        )

        if return_stage_maps:
            return (
                tokens,
                stem_map,
                stage1_map,
                stage2_map,
                patch_map,
            )

        if return_maps:
            return (
                tokens,
                stem_map,
                patch_map,
            )

        return tokens


# ============================================================
# D4 Dynamic Hierarchical Model
# ============================================================
class DTFHybridModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        stem_dim: int = 64,
        d_model: int = 256,
        freq_patches: int = 16,
        time_patches: int = 100,
        time_depth: int = 1,
        freq_depth: int = 1,
        num_heads: int = 8,
        dropout: float = 0.15,
        head_dropout: float = 0.20,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        binary_residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        if num_classes != 4:
            raise ValueError(
                "当前模型只支持ICBHI四分类。"
            )

        self.num_classes = num_classes

        self.binary_residual_scale = float(
            binary_residual_scale
        )

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

        self.four_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(
                d_model,
                4,
            ),
        )

        self.binary_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(
                d_model,
                2,
            ),
        )

        self.abnormal_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(
                d_model,
                3,
            ),
        )

    def get_dtf_alpha(
        self,
    ) -> float:
        return self.frontend.get_alpha()

    def extract_feature(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.frontend(x)
        feature = self.encoder(tokens)

        return feature

    @staticmethod
    def build_four_binary_logits(
        four_logits: torch.Tensor,
    ) -> torch.Tensor:
        normal_logit = (
            four_logits[:, 0]
        )

        abnormal_logit = torch.logsumexp(
            four_logits[:, 1:4],
            dim=1,
        )

        return torch.stack(
            [
                normal_logit,
                abnormal_logit,
            ],
            dim=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feature = self.extract_feature(x)

        four_logits = self.four_head(
            feature
        )

        four_binary_logits = (
            self.build_four_binary_logits(
                four_logits
            )
        )

        binary_residual_logits = (
            self.binary_head(feature)
        )

        binary_logits = (
            four_binary_logits
            + self.binary_residual_scale
            * binary_residual_logits
        )

        abnormal_logits = (
            self.abnormal_head(feature)
        )

        return {
            "feature": feature,

            "four_logits": four_logits,

            "four_binary_logits": (
                four_binary_logits
            ),

            "binary_residual_logits": (
                binary_residual_logits
            ),

            "binary_logits": binary_logits,

            "abnormal_logits": (
                abnormal_logits
            ),
        }

    @staticmethod
    def build_probabilities(
        outputs: Dict[str, torch.Tensor],
        four_weight: Optional[float] = None,
        minimum_hierarchical_weight: float = 0.15,
        maximum_hierarchical_weight: float = 0.60,
    ) -> Dict[str, torch.Tensor]:
        four_probability = torch.softmax(
            outputs["four_logits"],
            dim=1,
        )

        binary_probability = torch.softmax(
            outputs["binary_logits"],
            dim=1,
        )

        abnormal_probability = torch.softmax(
            outputs["abnormal_logits"],
            dim=1,
        )

        normal_probability = (
            binary_probability[:, 0:1]
        )

        hierarchical_abnormal_probability = (
            binary_probability[:, 1:2]
            * abnormal_probability
        )

        hierarchical_probability = torch.cat(
            [
                normal_probability,
                hierarchical_abnormal_probability,
            ],
            dim=1,
        )

        binary_entropy = -torch.sum(
            binary_probability
            * torch.log(
                binary_probability.clamp_min(
                    1e-8
                )
            ),
            dim=1,
            keepdim=True,
        ) / math.log(2.0)

        binary_confidence = (
            1.0
            - binary_entropy
        ).clamp(
            min=0.0,
            max=1.0,
        )

        if four_weight is None:
            hierarchical_weight = (
                minimum_hierarchical_weight
                + (
                    maximum_hierarchical_weight
                    - minimum_hierarchical_weight
                )
                * binary_confidence
            )

            four_probability_weight = (
                1.0
                - hierarchical_weight
            )

        else:
            if not 0.0 <= four_weight <= 1.0:
                raise ValueError(
                    "four_weight必须位于[0,1]。"
                )

            batch_size = (
                four_probability.shape[0]
            )

            four_probability_weight = (
                four_probability.new_full(
                    (batch_size, 1),
                    float(four_weight),
                )
            )

            hierarchical_weight = (
                1.0
                - four_probability_weight
            )

        final_probability = (
            four_probability_weight
            * four_probability
            + hierarchical_weight
            * hierarchical_probability
        )

        final_probability = (
            final_probability
            / final_probability.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        return {
            "four_probability": (
                four_probability
            ),

            "binary_probability": (
                binary_probability
            ),

            "abnormal_probability": (
                abnormal_probability
            ),

            "hierarchical_probability": (
                hierarchical_probability
            ),

            "binary_entropy": (
                binary_entropy
            ),

            "binary_confidence": (
                binary_confidence
            ),

            "four_probability_weight": (
                four_probability_weight
            ),

            "hierarchical_weight": (
                hierarchical_weight
            ),

            "final_probability": (
                final_probability
            ),
        }


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
        freq_patches=16,
        time_patches=100,
        time_depth=1,
        freq_depth=1,
        num_heads=8,
        dropout=0.15,
        head_dropout=0.20,
        d_state=16,
        d_conv=4,
        expand=2,
        binary_residual_scale=0.50,
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
            stage1_map,
            stage2_map,
            patch_map,
        ) = model.frontend(
            dummy_input,
            return_stage_maps=True,
        )

        outputs = model(dummy_input)

        probabilities = (
            model.build_probabilities(
                outputs,
                four_weight=None,
                minimum_hierarchical_weight=0.15,
                maximum_hierarchical_weight=0.60,
            )
        )

    print(
        "Input:",
        tuple(dummy_input.shape),
    )

    print(
        "Stem:",
        tuple(stem_map.shape),
    )

    print(
        "Stage1:",
        tuple(stage1_map.shape),
    )

    print(
        "Stage2:",
        tuple(stage2_map.shape),
    )

    print(
        "Patch:",
        tuple(patch_map.shape),
    )

    print(
        "Tokens:",
        tuple(tokens.shape),
    )

    print(
        "Feature:",
        tuple(
            outputs["feature"].shape
        ),
    )

    print(
        "Four logits:",
        tuple(
            outputs["four_logits"].shape
        ),
    )

    print(
        "Binary logits:",
        tuple(
            outputs["binary_logits"].shape
        ),
    )

    print(
        "Abnormal logits:",
        tuple(
            outputs["abnormal_logits"].shape
        ),
    )

    print(
        "Final probability:",
        tuple(
            probabilities[
                "final_probability"
            ].shape
        ),
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
        stage1_map.shape
    ) == (
        2,
        96,
        200,
        32,
    )

    assert tuple(
        stage2_map.shape
    ) == (
        2,
        160,
        100,
        16,
    )

    assert tuple(
        patch_map.shape
    ) == (
        2,
        256,
        100,
        16,
    )

    assert tuple(
        tokens.shape
    ) == (
        2,
        1600,
        256,
    )

    assert tuple(
        outputs["four_logits"].shape
    ) == (
        2,
        4,
    )

    assert tuple(
        outputs["binary_logits"].shape
    ) == (
        2,
        2,
    )

    assert tuple(
        outputs["abnormal_logits"].shape
    ) == (
        2,
        3,
    )

    assert tuple(
        probabilities[
            "final_probability"
        ].shape
    ) == (
        2,
        4,
    )

    print(
        "D4 dynamic hierarchical model shape test passed."
    )