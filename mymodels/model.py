#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Mamba
# ============================================================
try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True

except Exception:
    Mamba = None
    HAS_MAMBA = False


# ============================================================
# 2. Feed Forward Network
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
        return self.network(x)


# ============================================================
# 3. Dynamic SAME Padding Conv2d
# ============================================================
class SamePadConv2d(nn.Module):
    """
    动态SAME Padding卷积。

    输入：
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
        input_time = int(
            x.shape[-2]
        )

        input_frequency = int(
            x.shape[-1]
        )

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
            input_time
            / stride_time
        )

        output_frequency = math.ceil(
            input_frequency
            / stride_frequency
        )

        effective_kernel_time = (
            dilation_time
            * (
                kernel_time
                - 1
            )
            + 1
        )

        effective_kernel_frequency = (
            dilation_frequency
            * (
                kernel_frequency
                - 1
            )
            + 1
        )

        total_padding_time = max(
            (
                output_time
                - 1
            )
            * stride_time
            + effective_kernel_time
            - input_time,
            0,
        )

        total_padding_frequency = max(
            (
                output_frequency
                - 1
            )
            * stride_frequency
            + effective_kernel_frequency
            - input_frequency,
            0,
        )

        padding_top = (
            total_padding_time
            // 2
        )

        padding_bottom = (
            total_padding_time
            - padding_top
        )

        padding_left = (
            total_padding_frequency
            // 2
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
# 4. Channel-Gated DTF Stem
# ============================================================
class ChannelGatedDTFStem(nn.Module):
    """
    样本级、通道级动态时频门控。

    输入：
        [B, 1, 798, 128]

    时间分支：
        kernel = (6, 3)

    频率分支：
        kernel = (3, 6)

    输出：
        [B, 64, 399, 64]

    与原D4区别：
        原D4所有样本、所有通道共享一个标量alpha。

        当前版本为每个样本和每个通道分别生成：
            time_gate
            frequency_gate

        并满足：
            time_gate + frequency_gate = 1
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
        gate_reduction: int = 4,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels

        self.time_branch = nn.Sequential(
            SamePadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=time_kernel,
                stride=(
                    2,
                    2,
                ),
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
                stride=(
                    1,
                    1,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),
        )

        self.frequency_branch = nn.Sequential(
            SamePadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=frequency_kernel,
                stride=(
                    2,
                    2,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),

            SamePadConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=frequency_kernel,
                stride=(
                    1,
                    1,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),
        )

        gate_hidden_channels = max(
            16,
            (
                out_channels
                * 2
            )
            // gate_reduction,
        )

        self.branch_gate = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 2,
                out_channels=gate_hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=gate_hidden_channels,
                out_channels=out_channels * 2,
                kernel_size=1,
                bias=True,
            ),
        )

        self.output_norm = nn.BatchNorm2d(
            out_channels
        )

        self.output_activation = nn.GELU()

        # 用于日志显示最近一次前向传播的平均时间门控权重
        self.register_buffer(
            "_last_time_gate_mean",
            torch.tensor(
                0.5,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self._initialize_gate()

    def _initialize_gate(
        self,
    ) -> None:
        final_layer = self.branch_gate[-1]

        if isinstance(
            final_layer,
            nn.Conv2d,
        ):
            nn.init.zeros_(
                final_layer.weight
            )

            if final_layer.bias is not None:
                nn.init.zeros_(
                    final_layer.bias
                )

    def get_alpha_tensor(
        self,
    ) -> torch.Tensor:
        """
        为兼容旧版D4日志接口，返回最近一次前向传播中
        时间分支门控权重的平均值。
        """

        return self._last_time_gate_mean

    def get_alpha(
        self,
    ) -> float:
        return float(
            self._last_time_gate_mean
            .detach()
            .cpu()
            .item()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_gates: bool = False,
    ):
        time_feature = self.time_branch(
            x
        )

        frequency_feature = (
            self.frequency_branch(
                x
            )
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

        batch_size = int(
            time_feature.shape[0]
        )

        # [B, 2C, T, F]
        combined_feature = torch.cat(
            [
                time_feature,
                frequency_feature,
            ],
            dim=1,
        )

        # [B, 2C, 1, 1]
        branch_descriptor = (
            F.adaptive_avg_pool2d(
                combined_feature,
                output_size=1,
            )
        )

        # [B, 2C, 1, 1]
        gate_logits = self.branch_gate(
            branch_descriptor
        )

        # [B, 2, C, 1, 1]
        gate_logits = gate_logits.reshape(
            batch_size,
            2,
            self.out_channels,
            1,
            1,
        )

        branch_gates = torch.softmax(
            gate_logits,
            dim=1,
        )

        time_gate = branch_gates[
            :,
            0,
        ]

        frequency_gate = branch_gates[
            :,
            1,
        ]

        fused_feature = (
            time_gate
            * time_feature
            + frequency_gate
            * frequency_feature
        )

        fused_feature = self.output_norm(
            fused_feature
        )

        fused_feature = self.output_activation(
            fused_feature
        )

        with torch.no_grad():
            self._last_time_gate_mean.copy_(
                time_gate.mean().detach()
            )

        if return_gates:
            return (
                fused_feature,
                time_gate,
                frequency_gate,
            )

        return fused_feature


# 兼容可能仍调用DTFStem的旧代码
DTFStem = ChannelGatedDTFStem


# ============================================================
# 5. Residual Convolution Block
# ============================================================
class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(
                channels
            ),
            nn.GELU(),

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(
                    3,
                    3,
                ),
                stride=(
                    1,
                    1,
                ),
                padding=(
                    1,
                    1,
                ),
                bias=False,
            ),

            nn.BatchNorm2d(
                channels
            ),
            nn.GELU(),
            nn.Dropout2d(
                dropout
            ),

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(
                    3,
                    3,
                ),
                stride=(
                    1,
                    1,
                ),
                padding=(
                    1,
                    1,
                ),
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.block(x)


# ============================================================
# 6. Progressive Downsampling
# ============================================================
class ProgressiveDownsample(nn.Module):
    """
    输入：
        [B, 64, 399, 64]

    Stage 1：
        [B, 96, 200, 32]

    Stage 2：
        [B, 160, 100, 16]

    Stage 3：
        [B, 256, 100, 16]
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
                in_channels=in_channels,
                out_channels=96,
                kernel_size=(
                    3,
                    3,
                ),
                stride=(
                    2,
                    2,
                ),
                padding=(
                    1,
                    1,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(
                96
            ),
            nn.GELU(),

            ResidualConvBlock(
                channels=96,
                dropout=dropout
                * 0.25,
            ),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=160,
                kernel_size=(
                    3,
                    3,
                ),
                stride=(
                    2,
                    2,
                ),
                padding=(
                    1,
                    1,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(
                160
            ),
            nn.GELU(),

            ResidualConvBlock(
                channels=160,
                dropout=dropout
                * 0.50,
            ),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(
                in_channels=160,
                out_channels=out_channels,
                kernel_size=(
                    3,
                    3,
                ),
                stride=(
                    1,
                    1,
                ),
                padding=(
                    1,
                    1,
                ),
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.GELU(),

            ResidualConvBlock(
                channels=out_channels,
                dropout=dropout,
            ),

            nn.Dropout2d(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stage_maps: bool = False,
    ):
        stage1_map = self.stage1(
            x
        )

        stage2_map = self.stage2(
            stage1_map
        )

        patch_map = self.stage3(
            stage2_map
        )

        if return_stage_maps:
            return (
                patch_map,
                stage1_map,
                stage2_map,
            )

        return patch_map


# ============================================================
# 7. Time-Mamba Block
# ============================================================
class TimeMambaBlock(nn.Module):
    """
    输入输出：
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
            # 未安装mamba_ssm时，仅用于形状测试
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
        normalized = self.norm1(
            x
        )

        if self.use_mamba:
            sequence_output = (
                self.sequence_model(
                    normalized
                )
            )

        else:
            sequence_output, _ = (
                self.sequence_model(
                    normalized
                )
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
                self.norm2(
                    x
                )
            )
        )

        return x


# ============================================================
# 8. Frequency-Attention Block
# ============================================================
class FrequencyAttentionBlock(nn.Module):
    """
    输入输出：
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
                f"dim={dim}不能被"
                f"num_heads={num_heads}整除。"
            )

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(
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
        normalized = self.norm1(
            x
        )

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
                self.norm2(
                    x
                )
            )
        )

        return x


# ============================================================
# 9. Parallel Time-Frequency Block
# ============================================================
class ParallelTimeFrequencyBlock(nn.Module):
    """
    输入：
        [B, F, T, D]

    并行结构：

                    ┌─ Time-Mamba ────────────┐
        Input ──────┤                         ├─ Gate Fusion
                    └─ Frequency-Attention ───┘

    时间和频率分支从同一个输入开始，避免串联结构中
    前一分支过度修改后一分支的输入表示。
    """

    def __init__(
        self,
        dim: int,
        time_depth: int = 1,
        freq_depth: int = 1,
        num_heads: int = 8,
        dropout: float = 0.15,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()

        if time_depth < 1:
            raise ValueError(
                "time_depth必须至少为1。"
            )

        if freq_depth < 1:
            raise ValueError(
                "freq_depth必须至少为1。"
            )

        self.dim = dim

        self.time_blocks = nn.ModuleList(
            [
                TimeMambaBlock(
                    dim=dim,
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

        self.frequency_blocks = nn.ModuleList(
            [
                FrequencyAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(
                    freq_depth
                )
            ]
        )

        self.gate_norm = nn.LayerNorm(
            dim * 2
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(
                dim * 2,
                dim,
            ),
            nn.Sigmoid(),
        )

        self.fusion_projection = nn.Linear(
            dim,
            dim,
        )

        self.fusion_dropout = nn.Dropout(
            dropout
        )

        self.output_norm = nn.LayerNorm(
            dim
        )

        self.output_ffn = FeedForward(
            dim=dim,
            hidden_dim=dim * 2,
            dropout=dropout,
        )

        self.register_buffer(
            "_last_time_gate_mean",
            torch.tensor(
                0.5,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def get_time_gate_mean(
        self,
    ) -> float:
        return float(
            self._last_time_gate_mean
            .detach()
            .cpu()
            .item()
        )

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(
                "ParallelTimeFrequencyBlock输入"
                "必须为[B,F,T,D]，"
                f"当前={tuple(x.shape)}。"
            )

        (
            batch_size,
            frequency_patches,
            time_patches,
            dim,
        ) = x.shape

        if dim != self.dim:
            raise ValueError(
                f"输入维度错误：当前={dim}，"
                f"要求={self.dim}。"
            )

        residual = x

        # ----------------------------------------------------
        # Time-Mamba Branch
        # [B,F,T,D] -> [B*F,T,D]
        # ----------------------------------------------------
        time_sequence = x.reshape(
            batch_size
            * frequency_patches,
            time_patches,
            dim,
        )

        for block in self.time_blocks:
            time_sequence = block(
                time_sequence
            )

        time_output = time_sequence.reshape(
            batch_size,
            frequency_patches,
            time_patches,
            dim,
        )

        # ----------------------------------------------------
        # Frequency-Attention Branch
        # [B,F,T,D] -> [B*T,F,D]
        # ----------------------------------------------------
        frequency_sequence = x.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        frequency_sequence = (
            frequency_sequence.reshape(
                batch_size
                * time_patches,
                frequency_patches,
                dim,
            )
        )

        for block in self.frequency_blocks:
            frequency_sequence = block(
                frequency_sequence
            )

        frequency_output = (
            frequency_sequence.reshape(
                batch_size,
                time_patches,
                frequency_patches,
                dim,
            )
        )

        frequency_output = (
            frequency_output.permute(
                0,
                2,
                1,
                3,
            ).contiguous()
        )

        # ----------------------------------------------------
        # 融合时间与频率分支
        # ----------------------------------------------------
        gate_input = torch.cat(
            [
                time_output,
                frequency_output,
            ],
            dim=-1,
        )

        gate_input = self.gate_norm(
            gate_input
        )

        # gate接近1：更多使用时间分支
        # gate接近0：更多使用频率分支
        time_gate = self.fusion_gate(
            gate_input
        )

        fused_feature = (
            time_gate
            * time_output
            + (
                1.0
                - time_gate
            )
            * frequency_output
        )

        fused_feature = self.fusion_projection(
            fused_feature
        )

        # 使用残差连接提高训练稳定性
        x = (
            residual
            + self.fusion_dropout(
                fused_feature
                - residual
            )
        )

        x = self.output_norm(
            x
        )

        x = (
            x
            + self.output_ffn(
                x
            )
        )

        with torch.no_grad():
            self._last_time_gate_mean.copy_(
                time_gate.mean().detach()
            )

        if return_gate:
            return (
                x,
                time_gate,
            )

        return x


# ============================================================
# 10. Attentive Statistics Pooling
# ============================================================
class AttentiveStatisticsPooling(nn.Module):
    """
    使用注意力加权均值和标准差聚合Token。

    输入：
        [B, N, D]

    输出：
        [B, D]

    相比Max Pooling：
        不容易被单个噪声峰值或短时高响应主导。
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        hidden_dim = max(
            64,
            dim // 2,
        )

        self.attention_score = nn.Sequential(
            nn.LayerNorm(
                dim
            ),
            nn.Linear(
                dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        self.statistics_fusion = nn.Sequential(
            nn.LayerNorm(
                dim * 2
            ),
            nn.Linear(
                dim * 2,
                dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.output_norm = nn.LayerNorm(
            dim
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "池化输入必须为[B,N,D]，"
                f"当前={tuple(tokens.shape)}。"
            )

        attention_logits = self.attention_score(
            tokens
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=1,
        )

        weighted_mean = torch.sum(
            attention_weights
            * tokens,
            dim=1,
        )

        centered_tokens = (
            tokens
            - weighted_mean.unsqueeze(
                1
            )
        )

        weighted_variance = torch.sum(
            attention_weights
            * centered_tokens.pow(
                2
            ),
            dim=1,
        )

        weighted_std = torch.sqrt(
            weighted_variance.clamp_min(
                1e-5
            )
        )

        statistics = torch.cat(
            [
                weighted_mean,
                weighted_std,
            ],
            dim=-1,
        )

        feature = self.statistics_fusion(
            statistics
        )

        feature = self.output_norm(
            feature
        )

        return feature


# ============================================================
# 11. Parallel Time-Frequency Encoder
# ============================================================
class TimeFrequencyEncoder(nn.Module):
    """
    输入：
        [B, 1600, 256]

    网格：
        [B, 16, 100, 256]

    两个并行时频块：
        Time-Mamba || Frequency-Attention

    输出：
        [B, 256]
    """

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
        num_tf_blocks: int = 2,
    ) -> None:
        super().__init__()

        if freq_patches <= 0:
            raise ValueError(
                "freq_patches必须大于0。"
            )

        if time_patches <= 0:
            raise ValueError(
                "time_patches必须大于0。"
            )

        if num_tf_blocks <= 0:
            raise ValueError(
                "num_tf_blocks必须大于0。"
            )

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

        self.num_tf_blocks = (
            num_tf_blocks
        )

        self.input_projection = nn.Sequential(
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

        self.tf_blocks = nn.ModuleList(
            [
                ParallelTimeFrequencyBlock(
                    dim=d_model,
                    time_depth=time_depth,
                    freq_depth=freq_depth,
                    num_heads=num_heads,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(
                    num_tf_blocks
                )
            ]
        )

        self.final_grid_norm = nn.LayerNorm(
            d_model
        )

        self.pooling = (
            AttentiveStatisticsPooling(
                dim=d_model,
                dropout=dropout,
            )
        )

    def get_tf_gate_means(
        self,
    ):
        return [
            block.get_time_gate_mean()
            for block in self.tf_blocks
        ]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Encoder输入必须为[B,N,D]，"
                f"当前={tuple(x.shape)}。"
            )

        (
            batch_size,
            num_tokens,
            input_dim,
        ) = x.shape

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

        x = self.input_projection(
            x
        )

        # [B,F*T,D] -> [B,F,T,D]
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

        x = self.position_dropout(
            x
        )

        for block in self.tf_blocks:
            x = block(
                x
            )

        x = self.final_grid_norm(
            x
        )

        # [B,F,T,D] -> [B,F*T,D]
        tokens = x.reshape(
            batch_size,
            self.num_tokens,
            self.d_model,
        )

        feature = self.pooling(
            tokens
        )

        return feature


# ============================================================
# 12. DTF Frontend
# ============================================================
class DTFFrontend(nn.Module):
    """
    输入：
        [B,1,798,128]

    Channel-Gated DTF Stem：
        [B,64,399,64]

    Progressive Downsampling：
        [B,256,100,16]

    Tokens：
        [B,1600,256]
    """

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

        self.stem = ChannelGatedDTFStem(
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
                f"当前={tuple(x.shape)}。"
            )

        if x.shape[1] != 1:
            raise ValueError(
                "输入通道数必须为1，"
                f"当前={x.shape[1]}。"
            )

        if tuple(
            x.shape[-2:]
        ) != (
            798,
            128,
        ):
            raise ValueError(
                "Fbank尺寸必须为[798,128]，"
                f"当前={tuple(x.shape[-2:])}。"
            )

        stem_map = self.stem(
            x
        )

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
                f"要求空间尺寸={expected_map_shape}。"
            )

        batch_size = int(
            patch_map.shape[0]
        )

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
# 13. D4.2 Parallel Hybrid Model
# ============================================================
class DTFHybridModel(nn.Module):
    """
    D4.2架构：

        Fbank
          ↓
        Channel-Gated DTF Stem
          ↓
        Progressive Downsampling
          ↓
        Parallel Time-Mamba / Frequency-Attention × 2
          ↓
        Attentive Statistics Pooling
          ↓
        Shared Feature
          ├── Four-Class Head
          ├── Binary Residual Head
          └── Abnormal Subtype Head
          ↓
        Dynamic or Fixed Hierarchical Fusion
    """

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
        binary_residual_scale: float = 0.50,
        num_tf_blocks: int = 2,
    ) -> None:
        super().__init__()

        if num_classes != 4:
            raise ValueError(
                "当前模型仅支持ICBHI四分类。"
            )

        self.num_classes = num_classes
        self.d_model = d_model

        self.binary_residual_scale = float(
            binary_residual_scale
        )

        self.num_tf_blocks = int(
            num_tf_blocks
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
            num_tf_blocks=num_tf_blocks,
        )

        self.four_head = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Dropout(
                head_dropout
            ),
            nn.Linear(
                d_model,
                4,
            ),
        )

        self.binary_head = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Dropout(
                head_dropout
            ),
            nn.Linear(
                d_model,
                2,
            ),
        )

        self.abnormal_head = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),
            nn.Dropout(
                head_dropout
            ),
            nn.Linear(
                d_model,
                3,
            ),
        )

        self._initialize_heads()

    def _initialize_heads(
        self,
    ) -> None:
        for head in [
            self.four_head,
            self.binary_head,
            self.abnormal_head,
        ]:
            for module in head.modules():
                if isinstance(
                    module,
                    nn.Linear,
                ):
                    nn.init.trunc_normal_(
                        module.weight,
                        std=0.02,
                    )

                    if module.bias is not None:
                        nn.init.zeros_(
                            module.bias
                        )

    def get_dtf_alpha(
        self,
    ) -> float:
        """
        与旧版train.py兼容。

        当前返回Channel-Gated DTF Stem中
        时间分支门控的平均值。
        """

        return self.frontend.get_alpha()

    def get_tf_gate_means(
        self,
    ):
        return self.encoder.get_tf_gate_means()

    def extract_feature(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.frontend(
            x
        )

        feature = self.encoder(
            tokens
        )

        return feature

    @staticmethod
    def build_four_binary_logits(
        four_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        将四分类Logits聚合为：

            Normal
            Abnormal
        """

        normal_logit = four_logits[
            :,
            0,
        ]

        abnormal_logit = torch.logsumexp(
            four_logits[
                :,
                1:4,
            ],
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
        feature = self.extract_feature(
            x
        )

        four_logits = self.four_head(
            feature
        )

        four_binary_logits = (
            self.build_four_binary_logits(
                four_logits
            )
        )

        binary_residual_logits = (
            self.binary_head(
                feature
            )
        )

        binary_logits = (
            four_binary_logits
            + self.binary_residual_scale
            * binary_residual_logits
        )

        abnormal_logits = self.abnormal_head(
            feature
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

            "binary_logits": (
                binary_logits
            ),

            "abnormal_logits": (
                abnormal_logits
            ),
        }

    @staticmethod
    def build_probabilities(
        outputs: Dict[str, torch.Tensor],
        four_weight: Optional[float] = None,
        minimum_hierarchical_weight: float = 0.05,
        maximum_hierarchical_weight: float = 0.25,
    ) -> Dict[str, torch.Tensor]:
        """
        支持两种融合方式。

        1. four_weight=None：
           根据Binary Head置信度动态计算层级权重。

        2. four_weight为固定值：
           例如four_weight=0.85，
           表示四分类概率权重为0.85，
           层级概率权重为0.15。
        """

        if not (
            0.0
            <= minimum_hierarchical_weight
            <= maximum_hierarchical_weight
            <= 1.0
        ):
            raise ValueError(
                "必须满足："
                "0 <= min_hierarchical_weight "
                "<= max_hierarchical_weight <= 1。"
            )

        four_probability = torch.softmax(
            outputs[
                "four_logits"
            ],
            dim=1,
        )

        binary_probability = torch.softmax(
            outputs[
                "binary_logits"
            ],
            dim=1,
        )

        abnormal_probability = torch.softmax(
            outputs[
                "abnormal_logits"
            ],
            dim=1,
        )

        hierarchical_normal_probability = (
            binary_probability[
                :,
                0:1,
            ]
        )

        hierarchical_abnormal_probability = (
            binary_probability[
                :,
                1:2,
            ]
            * abnormal_probability
        )

        hierarchical_probability = torch.cat(
            [
                hierarchical_normal_probability,
                hierarchical_abnormal_probability,
            ],
            dim=1,
        )

        hierarchical_probability = (
            hierarchical_probability
            / hierarchical_probability.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(
                1e-8
            )
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
        )

        binary_entropy = (
            binary_entropy
            / math.log(
                2.0
            )
        )

        binary_entropy = binary_entropy.clamp(
            min=0.0,
            max=1.0,
        )

        binary_confidence = (
            1.0
            - binary_entropy
        ).clamp(
            min=0.0,
            max=1.0,
        )

        batch_size = int(
            four_probability.shape[0]
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
            four_weight = float(
                four_weight
            )

            if not (
                0.0
                <= four_weight
                <= 1.0
            ):
                raise ValueError(
                    "four_weight必须位于[0,1]。"
                )

            four_probability_weight = (
                four_probability.new_full(
                    (
                        batch_size,
                        1,
                    ),
                    four_weight,
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
            ).clamp_min(
                1e-8
            )
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
# 14. Shape Test
# ============================================================
if __name__ == "__main__":
    print(
        "HAS_MAMBA =",
        HAS_MAMBA,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device =",
        device,
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

        num_tf_blocks=2,
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
            stage1_map,
            stage2_map,
            patch_map,
        ) = model.frontend(
            dummy_input,
            return_stage_maps=True,
        )

        outputs = model(
            dummy_input
        )

        probabilities = (
            model.build_probabilities(
                outputs=outputs,

                four_weight=None,

                minimum_hierarchical_weight=0.05,

                maximum_hierarchical_weight=0.25,
            )
        )

    print(
        "Input:",
        tuple(
            dummy_input.shape
        ),
    )

    print(
        "Channel-Gated DTF Stem:",
        tuple(
            stem_map.shape
        ),
    )

    print(
        "Progressive Stage 1:",
        tuple(
            stage1_map.shape
        ),
    )

    print(
        "Progressive Stage 2:",
        tuple(
            stage2_map.shape
        ),
    )

    print(
        "Progressive Patch Map:",
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
        "Global Feature:",
        tuple(
            outputs[
                "feature"
            ].shape
        ),
    )

    print(
        "Four Logits:",
        tuple(
            outputs[
                "four_logits"
            ].shape
        ),
    )

    print(
        "Binary Logits:",
        tuple(
            outputs[
                "binary_logits"
            ].shape
        ),
    )

    print(
        "Abnormal Logits:",
        tuple(
            outputs[
                "abnormal_logits"
            ].shape
        ),
    )

    print(
        "Final Probability:",
        tuple(
            probabilities[
                "final_probability"
            ].shape
        ),
    )

    print(
        "DTF Time Gate Mean:",
        model.get_dtf_alpha(),
    )

    print(
        "Parallel TF Time Gate Means:",
        model.get_tf_gate_means(),
    )

    print(
        "Mean Hierarchical Weight:",
        float(
            probabilities[
                "hierarchical_weight"
            ].mean()
            .detach()
            .cpu()
            .item()
        ),
    )

    # --------------------------------------------------------
    # Shape Assertions
    # --------------------------------------------------------
    assert tuple(
        dummy_input.shape
    ) == (
        2,
        1,
        798,
        128,
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
        outputs[
            "feature"
        ].shape
    ) == (
        2,
        256,
    )

    assert tuple(
        outputs[
            "four_logits"
        ].shape
    ) == (
        2,
        4,
    )

    assert tuple(
        outputs[
            "binary_logits"
        ].shape
    ) == (
        2,
        2,
    )

    assert tuple(
        outputs[
            "abnormal_logits"
        ].shape
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

    probability_sum = probabilities[
        "final_probability"
    ].sum(
        dim=1
    )

    assert torch.allclose(
        probability_sum,
        torch.ones_like(
            probability_sum
        ),
        atol=1e-5,
    )

    print(
        "D4.2 parallel time-frequency model shape test passed."
    )