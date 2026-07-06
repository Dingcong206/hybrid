#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)


# ============================================================
# Project Import
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from mymodels.model import (
    HAS_MAMBA,
    DTFHybridModel,
)


# ============================================================
# Configuration
# ============================================================
CONFIG: Dict[str, object] = {
    # --------------------------------------------------------
    # 数据目录
    # --------------------------------------------------------
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_fbank"
    ),

    # --------------------------------------------------------
    # 保存目录
    # --------------------------------------------------------
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_d3_hierarchical_multitask_seed42"
    ),

    # --------------------------------------------------------
    # 训练设置
    # --------------------------------------------------------
    "EPOCHS": 50,

    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 4,

    "SEED": 42,

    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # --------------------------------------------------------
    # Fbank尺寸
    # --------------------------------------------------------
    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

    # --------------------------------------------------------
    # 模型结构
    # --------------------------------------------------------
    "STEM_DIM": 64,
    "D_MODEL": 256,

    "FREQ_PATCHES": 16,
    "TIME_PATCHES": 100,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,

    "D_STATE": 16,
    "D_CONV": 4,
    "EXPAND": 2,

    "DROPOUT": 0.15,
    "HEAD_DROPOUT": 0.20,

    # --------------------------------------------------------
    # 多任务损失
    #
    # Total Loss =
    # 1.0 * Four Loss
    # + 0.5 * Binary Loss
    # + 0.5 * Abnormal Loss
    # --------------------------------------------------------
    "FOUR_LOSS_WEIGHT": 1.0,
    "BINARY_LOSS_WEIGHT": 0.5,
    "ABNORMAL_LOSS_WEIGHT": 0.5,

    # --------------------------------------------------------
    # 推理融合
    #
    # Final Probability =
    # 0.5 * Four-class Probability
    # + 0.5 * Hierarchical Probability
    # --------------------------------------------------------
    "FOUR_PROBABILITY_WEIGHT": 0.5,

    # --------------------------------------------------------
    # 类别权重
    #
    # 第一轮层级实验暂时关闭类别权重
    # --------------------------------------------------------
    "USE_FOUR_CLASS_WEIGHTS": False,
    "USE_BINARY_CLASS_WEIGHTS": False,
    "USE_ABNORMAL_CLASS_WEIGHTS": False,

    "CLASS_WEIGHT_POWER": 0.25,
    "CLASS_WEIGHT_MAX": 1.50,

    # --------------------------------------------------------
    # Label Smoothing
    # --------------------------------------------------------
    "FOUR_LABEL_SMOOTHING": 0.0,
    "BINARY_LABEL_SMOOTHING": 0.0,
    "ABNORMAL_LABEL_SMOOTHING": 0.0,

    # --------------------------------------------------------
    # SpecAugment
    # --------------------------------------------------------
    "USE_SPECAUGMENT": True,

    "TIME_MASK_MAX": 80,
    "FREQ_MASK_MAX": 16,

    # --------------------------------------------------------
    # 学习率
    # --------------------------------------------------------
    "FRONTEND_LR": 3e-4,
    "ENCODER_LR": 1e-4,
    "HEAD_LR": 3e-4,

    "MIN_FRONTEND_LR": 3e-6,
    "MIN_ENCODER_LR": 1e-6,
    "MIN_HEAD_LR": 3e-6,

    "WARMUP_EPOCHS": 3,

    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # --------------------------------------------------------
    # 日志
    # --------------------------------------------------------
    "PRINT_INTERVAL": 50,
}


# ============================================================
# Random Seed
# ============================================================
def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except AttributeError:
        pass


# ============================================================
# AMP GradScaler
# ============================================================
def make_scaler(
    enabled: bool,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (
        AttributeError,
        TypeError,
    ):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


# ============================================================
# Warmup + Cosine Learning Rate
# ============================================================
def set_epoch_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs,
    min_lrs,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
):
    if epoch <= warmup_epochs:
        warmup_scale = (
            0.20
            + 0.80
            * epoch
            / max(
                warmup_epochs,
                1,
            )
        )

        current_lrs = [
            base_lr * warmup_scale
            for base_lr in base_lrs
        ]

    else:
        cosine_total = max(
            total_epochs
            - warmup_epochs,
            1,
        )

        cosine_step = min(
            epoch
            - warmup_epochs,
            cosine_total,
        )

        cosine_ratio = 0.5 * (
            1.0
            + math.cos(
                math.pi
                * cosine_step
                / cosine_total
            )
        )

        current_lrs = [
            min_lr
            + (
                base_lr
                - min_lr
            )
            * cosine_ratio
            for base_lr, min_lr
            in zip(
                base_lrs,
                min_lrs,
            )
        ]

    for parameter_group, current_lr in zip(
        optimizer.param_groups,
        current_lrs,
    ):
        parameter_group["lr"] = float(
            current_lr
        )

    return current_lrs


# ============================================================
# SpecAugment
# ============================================================
def apply_specaugment(
    fbank: torch.Tensor,
    time_mask_max: int,
    frequency_mask_max: int,
) -> torch.Tensor:
    """
    输入：
        [T, F]

    输出：
        [T, F]
    """

    x = fbank.clone()

    time_frames = int(
        x.shape[0]
    )

    frequency_bins = int(
        x.shape[1]
    )

    mask_value = x.mean()

    # --------------------------------------------------------
    # Time Mask
    # --------------------------------------------------------
    if time_mask_max > 0:
        time_width = random.randint(
            0,
            min(
                time_mask_max,
                time_frames,
            ),
        )

        if time_width > 0:
            time_start = random.randint(
                0,
                time_frames
                - time_width,
            )

            x[
                time_start:
                time_start + time_width,
                :
            ] = mask_value

    # --------------------------------------------------------
    # Frequency Mask
    # --------------------------------------------------------
    if frequency_mask_max > 0:
        frequency_width = random.randint(
            0,
            min(
                frequency_mask_max,
                frequency_bins,
            ),
        )

        if frequency_width > 0:
            frequency_start = random.randint(
                0,
                frequency_bins
                - frequency_width,
            )

            x[
                :,
                frequency_start:
                frequency_start + frequency_width,
            ] = mask_value

    return x


# ============================================================
# Fbank Dataset
# ============================================================
class FbankDataset(Dataset):
    """
    CSV必须包含：

        fbank_path
        label

    单个Fbank：

        [798, 128]

    返回：

        x: [1, 798, 128]
        y: Long Tensor
    """

    def __init__(
        self,
        csv_path,
        cfg,
        training: bool = False,
    ) -> None:
        super().__init__()

        self.csv_path = Path(
            csv_path
        )

        self.cfg = cfg
        self.training = training

        self.expected_shape = (
            int(
                cfg[
                    "FBANK_FRAMES"
                ]
            ),
            int(
                cfg[
                    "FBANK_MELS"
                ]
            ),
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV文件不存在："
                f"{self.csv_path}"
            )

        self.dataframe = pd.read_csv(
            self.csv_path
        ).reset_index(
            drop=True
        )

        required_columns = {
            "fbank_path",
            "label",
        }

        missing_columns = (
            required_columns
            - set(
                self.dataframe.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{self.csv_path}缺少列："
                f"{sorted(missing_columns)}"
            )

        self.dataframe[
            "label"
        ] = self.dataframe[
            "label"
        ].astype(
            int
        )

        self.labels = self.dataframe[
            "label"
        ].to_numpy(
            dtype=np.int64
        )

        invalid_labels = np.unique(
            self.labels[
                (
                    self.labels < 0
                )
                |
                (
                    self.labels > 3
                )
            ]
        )

        if len(
            invalid_labels
        ) > 0:
            raise ValueError(
                "发现非法标签："
                f"{invalid_labels.tolist()}"
            )

        self.class_counts = np.bincount(
            self.labels,
            minlength=4,
        )

        print(
            f"[FbankDataset] "
            f"samples={len(self.dataframe)} | "
            f"counts={self.class_counts.tolist()} | "
            f"shape={self.expected_shape} | "
            f"training={self.training} | "
            f"csv={self.csv_path}",
            flush=True,
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.dataframe
        )

    def resolve_path(
        self,
        raw_path,
    ) -> Path:
        fbank_path = Path(
            str(
                raw_path
            )
        )

        if fbank_path.exists():
            return fbank_path

        relative_path = (
            self.csv_path.parent
            / fbank_path
        )

        if relative_path.exists():
            return relative_path

        raise FileNotFoundError(
            f"Fbank文件不存在："
            f"{raw_path}"
        )

    def __getitem__(
        self,
        index,
    ):
        row = self.dataframe.iloc[
            index
        ]

        fbank_path = self.resolve_path(
            row[
                "fbank_path"
            ]
        )

        fbank = np.load(
            fbank_path,
            allow_pickle=False,
        )

        if tuple(
            fbank.shape
        ) != self.expected_shape:
            raise ValueError(
                f"Fbank尺寸错误："
                f"{fbank_path}\n"
                f"当前={tuple(fbank.shape)}，"
                f"要求={self.expected_shape}"
            )

        if not np.isfinite(
            fbank
        ).all():
            raise ValueError(
                "Fbank包含NaN或Inf："
                f"{fbank_path}"
            )

        x = torch.from_numpy(
            fbank
        ).float()

        if (
            self.training
            and bool(
                self.cfg[
                    "USE_SPECAUGMENT"
                ]
            )
        ):
            x = apply_specaugment(
                x,
                time_mask_max=int(
                    self.cfg[
                        "TIME_MASK_MAX"
                    ]
                ),
                frequency_mask_max=int(
                    self.cfg[
                        "FREQ_MASK_MAX"
                    ]
                ),
            )

        # [T,F] -> [1,T,F]
        x = x.unsqueeze(
            0
        )

        y = torch.tensor(
            int(
                row[
                    "label"
                ]
            ),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# DataLoader
# ============================================================
def make_loader(
    dataset,
    cfg,
    device,
    shuffle: bool,
):
    workers = int(
        cfg[
            "NUM_WORKERS"
        ]
    )

    loader_arguments = {
        "dataset": dataset,

        "batch_size": int(
            cfg[
                "BATCH_SIZE"
            ]
        ),

        "shuffle": shuffle,

        "num_workers": workers,

        "pin_memory": (
            device.type
            == "cuda"
        ),

        "persistent_workers": (
            workers > 0
        ),

        "drop_last": False,
    }

    if workers > 0:
        loader_arguments[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **loader_arguments
    )


# ============================================================
# Class Weight
# ============================================================
def make_class_weight(
    counts,
    use_weight: bool,
    power: float,
    maximum: float,
    device,
    name: str,
) -> Optional[torch.Tensor]:
    if not use_weight:
        print(
            f"[Loss] {name}不使用类别权重。",
            flush=True,
        )

        return None

    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    counts = np.maximum(
        counts,
        1.0,
    )

    weights = (
        counts.max()
        / counts
    ) ** power

    weights = (
        weights
        / weights.mean()
    )

    weights = np.minimum(
        weights,
        maximum,
    )

    weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[Loss] {name}权重：",
        np.round(
            weights,
            6,
        ).tolist(),
        flush=True,
    )

    return weight_tensor


def build_loss_weights(
    train_class_counts,
    cfg,
    device,
):
    four_counts = np.asarray(
        train_class_counts,
        dtype=np.int64,
    )

    binary_counts = np.asarray(
        [
            four_counts[0],
            four_counts[
                1:
            ].sum(),
        ],
        dtype=np.int64,
    )

    abnormal_counts = np.asarray(
        [
            four_counts[1],
            four_counts[2],
            four_counts[3],
        ],
        dtype=np.int64,
    )

    power = float(
        cfg[
            "CLASS_WEIGHT_POWER"
        ]
    )

    maximum = float(
        cfg[
            "CLASS_WEIGHT_MAX"
        ]
    )

    four_weight = make_class_weight(
        counts=four_counts,
        use_weight=bool(
            cfg[
                "USE_FOUR_CLASS_WEIGHTS"
            ]
        ),
        power=power,
        maximum=maximum,
        device=device,
        name="Four-class",
    )

    binary_weight = make_class_weight(
        counts=binary_counts,
        use_weight=bool(
            cfg[
                "USE_BINARY_CLASS_WEIGHTS"
            ]
        ),
        power=power,
        maximum=maximum,
        device=device,
        name="Binary",
    )

    abnormal_weight = make_class_weight(
        counts=abnormal_counts,
        use_weight=bool(
            cfg[
                "USE_ABNORMAL_CLASS_WEIGHTS"
            ]
        ),
        power=power,
        maximum=maximum,
        device=device,
        name="Abnormal",
    )

    return {
        "four": four_weight,
        "binary": binary_weight,
        "abnormal": abnormal_weight,
    }


# ============================================================
# Multi-task Loss
# ============================================================
def calculate_multitask_loss(
    outputs,
    labels,
    loss_weights,
    cfg,
):
    # --------------------------------------------------------
    # Four-class Loss
    #
    # 0 = Normal
    # 1 = Crackle
    # 2 = Wheeze
    # 3 = Both
    # --------------------------------------------------------
    four_loss = F.cross_entropy(
        outputs[
            "four_logits"
        ],
        labels,
        weight=loss_weights[
            "four"
        ],
        label_smoothing=float(
            cfg[
                "FOUR_LABEL_SMOOTHING"
            ]
        ),
    )

    # --------------------------------------------------------
    # Binary Loss
    #
    # 0 = Normal
    # 1 = Abnormal
    # --------------------------------------------------------
    binary_labels = (
        labels > 0
    ).long()

    binary_loss = F.cross_entropy(
        outputs[
            "binary_logits"
        ],
        binary_labels,
        weight=loss_weights[
            "binary"
        ],
        label_smoothing=float(
            cfg[
                "BINARY_LABEL_SMOOTHING"
            ]
        ),
    )

    # --------------------------------------------------------
    # Abnormal Subtype Loss
    #
    # 原始标签：
    # 1 = Crackle
    # 2 = Wheeze
    # 3 = Both
    #
    # 异常头标签：
    # 0 = Crackle
    # 1 = Wheeze
    # 2 = Both
    # --------------------------------------------------------
    abnormal_mask = (
        labels > 0
    )

    abnormal_count = int(
        abnormal_mask.sum().item()
    )

    if abnormal_count > 0:
        abnormal_labels = (
            labels[
                abnormal_mask
            ]
            - 1
        )

        abnormal_logits = outputs[
            "abnormal_logits"
        ][
            abnormal_mask
        ]

        abnormal_loss = F.cross_entropy(
            abnormal_logits,
            abnormal_labels,
            weight=loss_weights[
                "abnormal"
            ],
            label_smoothing=float(
                cfg[
                    "ABNORMAL_LABEL_SMOOTHING"
                ]
            ),
        )

    else:
        abnormal_loss = (
            outputs[
                "abnormal_logits"
            ].sum()
            * 0.0
        )

    total_loss = (
        float(
            cfg[
                "FOUR_LOSS_WEIGHT"
            ]
        )
        * four_loss
        + float(
            cfg[
                "BINARY_LOSS_WEIGHT"
            ]
        )
        * binary_loss
        + float(
            cfg[
                "ABNORMAL_LOSS_WEIGHT"
            ]
        )
        * abnormal_loss
    )

    return {
        "total_loss": total_loss,
        "four_loss": four_loss,
        "binary_loss": binary_loss,
        "abnormal_loss": abnormal_loss,
        "abnormal_count": abnormal_count,
    }


# ============================================================
# Train One Epoch
# ============================================================
def train_one_epoch(
    loader,
    model,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    loss_weights,
    cfg,
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_batches = len(
        loader
    )

    total_samples = 0
    total_abnormal_samples = 0

    accumulated_total_loss = 0.0
    accumulated_four_loss = 0.0
    accumulated_binary_loss = 0.0
    accumulated_abnormal_loss = 0.0

    epoch_start_time = time.time()

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    accum_steps = int(
        cfg[
            "ACCUM_STEPS"
        ]
    )

    print_interval = int(
        cfg[
            "PRINT_INTERVAL"
        ]
    )

    print(
        f"[TRAIN] "
        f"batches={total_batches} | "
        f"batch={cfg['BATCH_SIZE']} | "
        f"accum={accum_steps}",
        flush=True,
    )

    for batch_index, (
        x,
        y,
    ) in enumerate(
        loader
    ):
        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(
                x
            )

            loss_result = calculate_multitask_loss(
                outputs=outputs,
                labels=y,
                loss_weights=loss_weights,
                cfg=cfg,
            )

            total_loss = loss_result[
                "total_loss"
            ]

            backward_loss = (
                total_loss
                / accum_steps
            )

        scaler.scale(
            backward_loss
        ).backward()

        batch_size = int(
            y.shape[0]
        )

        abnormal_count = int(
            loss_result[
                "abnormal_count"
            ]
        )

        total_samples += batch_size
        total_abnormal_samples += abnormal_count

        accumulated_total_loss += (
            float(
                loss_result[
                    "total_loss"
                ].detach().item()
            )
            * batch_size
        )

        accumulated_four_loss += (
            float(
                loss_result[
                    "four_loss"
                ].detach().item()
            )
            * batch_size
        )

        accumulated_binary_loss += (
            float(
                loss_result[
                    "binary_loss"
                ].detach().item()
            )
            * batch_size
        )

        if abnormal_count > 0:
            accumulated_abnormal_loss += (
                float(
                    loss_result[
                        "abnormal_loss"
                    ].detach().item()
                )
                * abnormal_count
            )

        completed_batches = (
            batch_index + 1
        )

        should_update = (
            completed_batches
            % accum_steps
            == 0
            or completed_batches
            == total_batches
        )

        if should_update:
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(
                    cfg[
                        "GRAD_CLIP"
                    ]
                ),
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        if (
            completed_batches == 1
            or completed_batches
            % print_interval
            == 0
            or completed_batches
            == total_batches
        ):
            elapsed_time = (
                time.time()
                - epoch_start_time
            )

            average_batch_time = (
                elapsed_time
                / completed_batches
            )

            remaining_seconds = (
                total_batches
                - completed_batches
            ) * average_batch_time

            if device.type == "cuda":
                allocated_memory = (
                    torch.cuda
                    .memory_allocated(
                        device
                    )
                    / 1024**3
                )

                reserved_memory = (
                    torch.cuda
                    .memory_reserved(
                        device
                    )
                    / 1024**3
                )
            else:
                allocated_memory = 0.0
                reserved_memory = 0.0

            print(
                f"  Batch "
                f"{completed_batches:04d}/"
                f"{total_batches} | "
                f"Total "
                f"{loss_result['total_loss'].item():.4f} | "
                f"Four "
                f"{loss_result['four_loss'].item():.4f} | "
                f"Bin "
                f"{loss_result['binary_loss'].item():.4f} | "
                f"Abn "
                f"{loss_result['abnormal_loss'].item():.4f} | "
                f"ETA "
                f"{remaining_seconds / 60:.1f}min | "
                f"GPU "
                f"{allocated_memory:.2f}/"
                f"{reserved_memory:.2f}GB",
                flush=True,
            )

    average_total_loss = (
        accumulated_total_loss
        / max(
            total_samples,
            1,
        )
    )

    average_four_loss = (
        accumulated_four_loss
        / max(
            total_samples,
            1,
        )
    )

    average_binary_loss = (
        accumulated_binary_loss
        / max(
            total_samples,
            1,
        )
    )

    average_abnormal_loss = (
        accumulated_abnormal_loss
        / max(
            total_abnormal_samples,
            1,
        )
    )

    return {
        "total_loss": average_total_loss,
        "four_loss": average_four_loss,
        "binary_loss": average_binary_loss,
        "abnormal_loss": average_abnormal_loss,
    }


# ============================================================
# ICBHI Metrics
# ============================================================
def calculate_metrics(
    y_true,
    y_pred,
):
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
            3,
        ],
    )

    normal_total = max(
        int(
            confusion[
                0
            ].sum()
        ),
        1,
    )

    abnormal_total = max(
        int(
            confusion[
                1:
            ].sum()
        ),
        1,
    )

    specificity = (
        100.0
        * float(
            confusion[
                0,
                0,
            ]
        )
        / normal_total
    )

    sensitivity = (
        100.0
        * float(
            confusion[
                1,
                1,
            ]
            + confusion[
                2,
                2,
            ]
            + confusion[
                3,
                3,
            ]
        )
        / abnormal_total
    )

    score = (
        specificity
        + sensitivity
    ) / 2.0

    recalls = recall_score(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
            3,
        ],
        average=None,
        zero_division=0,
    )

    binary_true = (
        y_true > 0
    ).astype(
        np.int64
    )

    binary_pred = (
        y_pred > 0
    ).astype(
        np.int64
    )

    return {
        "score": float(
            score
        ),

        "sp": float(
            specificity
        ),

        "se": float(
            sensitivity
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
            * 100.0
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "recalls": recalls,

        "pred_counts": np.bincount(
            y_pred,
            minlength=4,
        ),

        "four_cm": confusion,

        "binary_cm": confusion_matrix(
            binary_true,
            binary_pred,
            labels=[
                0,
                1,
            ],
        ),
    }


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(
    loader,
    model,
    device,
    cfg,
    use_amp: bool,
):
    model.eval()

    all_labels = []

    all_final_predictions = []
    all_four_predictions = []
    all_hierarchical_predictions = []

    all_binary_head_predictions = []

    all_abnormal_true = []
    all_abnormal_head_predictions = []

    for x, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        y_device = y.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(
                x
            )

            probabilities = (
                model.build_probabilities(
                    outputs,
                    four_weight=float(
                        cfg[
                            "FOUR_PROBABILITY_WEIGHT"
                        ]
                    ),
                )
            )

        final_predictions = torch.argmax(
            probabilities[
                "final_probability"
            ],
            dim=1,
        )

        four_predictions = torch.argmax(
            probabilities[
                "four_probability"
            ],
            dim=1,
        )

        hierarchical_predictions = (
            torch.argmax(
                probabilities[
                    "hierarchical_probability"
                ],
                dim=1,
            )
        )

        binary_head_predictions = torch.argmax(
            outputs[
                "binary_logits"
            ],
            dim=1,
        )

        abnormal_head_predictions = torch.argmax(
            outputs[
                "abnormal_logits"
            ],
            dim=1,
        )

        all_labels.append(
            y.cpu()
        )

        all_final_predictions.append(
            final_predictions.cpu()
        )

        all_four_predictions.append(
            four_predictions.cpu()
        )

        all_hierarchical_predictions.append(
            hierarchical_predictions.cpu()
        )

        all_binary_head_predictions.append(
            binary_head_predictions.cpu()
        )

        abnormal_mask = (
            y_device > 0
        )

        if int(
            abnormal_mask.sum().item()
        ) > 0:
            all_abnormal_true.append(
                (
                    y_device[
                        abnormal_mask
                    ]
                    - 1
                ).cpu()
            )

            all_abnormal_head_predictions.append(
                abnormal_head_predictions[
                    abnormal_mask
                ].cpu()
            )

    y_true = torch.cat(
        all_labels
    ).numpy()

    final_pred = torch.cat(
        all_final_predictions
    ).numpy()

    four_pred = torch.cat(
        all_four_predictions
    ).numpy()

    hierarchical_pred = torch.cat(
        all_hierarchical_predictions
    ).numpy()

    binary_head_pred = torch.cat(
        all_binary_head_predictions
    ).numpy()

    binary_true = (
        y_true > 0
    ).astype(
        np.int64
    )

    binary_head_cm = confusion_matrix(
        binary_true,
        binary_head_pred,
        labels=[
            0,
            1,
        ],
    )

    binary_head_accuracy = (
        accuracy_score(
            binary_true,
            binary_head_pred,
        )
        * 100.0
    )

    if len(
        all_abnormal_true
    ) > 0:
        abnormal_true = torch.cat(
            all_abnormal_true
        ).numpy()

        abnormal_head_pred = torch.cat(
            all_abnormal_head_predictions
        ).numpy()

        abnormal_head_cm = confusion_matrix(
            abnormal_true,
            abnormal_head_pred,
            labels=[
                0,
                1,
                2,
            ],
        )

        abnormal_head_accuracy = (
            accuracy_score(
                abnormal_true,
                abnormal_head_pred,
            )
            * 100.0
        )

    else:
        abnormal_head_cm = np.zeros(
            (
                3,
                3,
            ),
            dtype=np.int64,
        )

        abnormal_head_accuracy = 0.0

    return {
        "final": calculate_metrics(
            y_true,
            final_pred,
        ),

        "four_only": calculate_metrics(
            y_true,
            four_pred,
        ),

        "hierarchical_only": calculate_metrics(
            y_true,
            hierarchical_pred,
        ),

        "binary_head_cm": (
            binary_head_cm
        ),

        "binary_head_accuracy": float(
            binary_head_accuracy
        ),

        "abnormal_head_cm": (
            abnormal_head_cm
        ),

        "abnormal_head_accuracy": float(
            abnormal_head_accuracy
        ),
    }


# ============================================================
# Shape Test
# ============================================================
@torch.no_grad()
def shape_test(
    loader,
    model,
    device,
) -> None:
    model.eval()

    x, y = next(
        iter(
            loader
        )
    )

    x = x[
        :2
    ].to(
        device
    )

    y = y[
        :2
    ].to(
        device
    )

    (
        tokens,
        stem_map,
        stage1_map,
        stage2_map,
        patch_map,
    ) = model.frontend(
        x,
        return_stage_maps=True,
    )

    outputs = model(
        x
    )

    probabilities = (
        model.build_probabilities(
            outputs,
            four_weight=0.5,
        )
    )

    print(
        "[Shape Test] Fbank:",
        tuple(
            x.shape
        ),
    )

    print(
        "[Shape Test] Stem Map:",
        tuple(
            stem_map.shape
        ),
    )

    print(
        "[Shape Test] Stage 1:",
        tuple(
            stage1_map.shape
        ),
    )

    print(
        "[Shape Test] Stage 2:",
        tuple(
            stage2_map.shape
        ),
    )

    print(
        "[Shape Test] Patch Map:",
        tuple(
            patch_map.shape
        ),
    )

    print(
        "[Shape Test] Tokens:",
        tuple(
            tokens.shape
        ),
    )

    print(
        "[Shape Test] Feature:",
        tuple(
            outputs[
                "feature"
            ].shape
        ),
    )

    print(
        "[Shape Test] Four Logits:",
        tuple(
            outputs[
                "four_logits"
            ].shape
        ),
    )

    print(
        "[Shape Test] Binary Logits:",
        tuple(
            outputs[
                "binary_logits"
            ].shape
        ),
    )

    print(
        "[Shape Test] Abnormal Logits:",
        tuple(
            outputs[
                "abnormal_logits"
            ].shape
        ),
    )

    print(
        "[Shape Test] Final Probability:",
        tuple(
            probabilities[
                "final_probability"
            ].shape
        ),
    )

    print(
        "[Shape Test] Labels:",
        tuple(
            y.shape
        ),
    )

    print(
        "[Shape Test] DTF Alpha:",
        model.get_dtf_alpha(),
    )

    assert tuple(
        x.shape[
            1:
        ]
    ) == (
        1,
        798,
        128,
    )

    assert tuple(
        stem_map.shape[
            1:
        ]
    ) == (
        64,
        399,
        64,
    )

    assert tuple(
        stage1_map.shape[
            1:
        ]
    ) == (
        96,
        200,
        32,
    )

    assert tuple(
        stage2_map.shape[
            1:
        ]
    ) == (
        160,
        100,
        16,
    )

    assert tuple(
        patch_map.shape[
            1:
        ]
    ) == (
        256,
        100,
        16,
    )

    assert tuple(
        tokens.shape[
            1:
        ]
    ) == (
        1600,
        256,
    )

    assert tuple(
        outputs[
            "feature"
        ].shape[
            1:
        ]
    ) == (
        256,
    )

    assert tuple(
        outputs[
            "four_logits"
        ].shape[
            1:
        ]
    ) == (
        4,
    )

    assert tuple(
        outputs[
            "binary_logits"
        ].shape[
            1:
        ]
    ) == (
        2,
    )

    assert tuple(
        outputs[
            "abnormal_logits"
        ].shape[
            1:
        ]
    ) == (
        3,
    )

    assert tuple(
        probabilities[
            "final_probability"
        ].shape[
            1:
        ]
    ) == (
        4,
    )

    print(
        "[PASS] 层级多任务模型连接成功。",
        flush=True,
    )

    model.train()


# ============================================================
# Print Metrics
# ============================================================
def print_metric_block(
    title: str,
    result,
) -> None:
    print()

    print(
        "-" * 80
    )

    print(
        title
    )

    print(
        "-" * 80
    )

    print(
        f"ICBHI Score: "
        f"{result['score']:.4f}"
    )

    print(
        f"Specificity: "
        f"{result['sp']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{result['se']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{result['macro_f1']:.4f}"
    )

    print(
        "Recall "
        "[Normal, Crackle, Wheeze, Both]:",
        np.round(
            result[
                "recalls"
            ],
            4,
        ).tolist(),
    )

    print(
        "PredCount:",
        result[
            "pred_counts"
        ].tolist(),
    )

    print()

    print(
        "Four-class confusion matrix:"
    )

    print(
        result[
            "four_cm"
        ]
    )

    print()

    print(
        "Binary confusion matrix:"
    )

    print(
        result[
            "binary_cm"
        ]
    )


def print_final(
    evaluation_result,
) -> None:
    print()

    print(
        "=" * 80
    )

    print(
        "FINAL OFFICIAL TEST RESULT"
    )

    print(
        "=" * 80
    )

    print_metric_block(
        "FINAL FUSED PREDICTION",
        evaluation_result[
            "final"
        ],
    )

    print_metric_block(
        "FOUR-CLASS HEAD ONLY",
        evaluation_result[
            "four_only"
        ],
    )

    print_metric_block(
        "HIERARCHICAL HEAD ONLY",
        evaluation_result[
            "hierarchical_only"
        ],
    )

    print()

    print(
        "-" * 80
    )

    print(
        "AUXILIARY HEAD RESULTS"
    )

    print(
        "-" * 80
    )

    print(
        f"Binary Head Accuracy: "
        f"{evaluation_result['binary_head_accuracy']:.4f}"
    )

    print(
        "Binary Head Confusion Matrix:"
    )

    print(
        evaluation_result[
            "binary_head_cm"
        ]
    )

    print()

    print(
        f"Abnormal Head Accuracy: "
        f"{evaluation_result['abnormal_head_accuracy']:.4f}"
    )

    print(
        "Abnormal Head Confusion Matrix "
        "[Crackle, Wheeze, Both]:"
    )

    print(
        evaluation_result[
            "abnormal_head_cm"
        ]
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    cfg = CONFIG

    set_seed(
        int(
            cfg[
                "SEED"
            ]
        )
    )

    device = torch.device(
        "cuda"
        if (
            str(
                cfg[
                    "DEVICE"
                ]
            ) == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    print(
        "[INFO] device:",
        device,
    )

    if device.type == "cuda":
        print(
            "[INFO] GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "[INFO] HAS_MAMBA:",
        HAS_MAMBA,
    )

    if (
        bool(
            cfg[
                "REQUIRE_MAMBA"
            ]
        )
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm导入失败，"
            "不能进行正式训练。"
        )

    root = Path(
        str(
            cfg[
                "ROOT"
            ]
        )
    )

    train_csv = (
        root
        / "train_index.csv"
    )

    test_csv = (
        root
        / "test_index.csv"
    )

    if not train_csv.exists():
        raise FileNotFoundError(
            train_csv
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            test_csv
        )

    print(
        "[Protocol] 使用完整官方训练集。"
    )

    print(
        "[Protocol] 不划分验证集。"
    )

    print(
        "[Protocol] 固定训练50轮，"
        "训练完成后测试一次。"
    )

    print(
        "[Input] 直接读取Fbank，"
        "不使用AST Token。"
    )

    print(
        "[Experiment] D3："
        "DTF Stem"
        " + Progressive Downsampling"
        " + Time-Mamba"
        " + Frequency-Attention"
        " + Hierarchical Multi-task Heads"
    )

    print(
        "[Loss Weight] "
        f"Four={cfg['FOUR_LOSS_WEIGHT']} | "
        f"Binary={cfg['BINARY_LOSS_WEIGHT']} | "
        f"Abnormal={cfg['ABNORMAL_LOSS_WEIGHT']}"
    )

    print(
        "[Probability Fusion] "
        f"Four={cfg['FOUR_PROBABILITY_WEIGHT']} | "
        f"Hierarchical="
        f"{1.0 - float(cfg['FOUR_PROBABILITY_WEIGHT'])}"
    )

    save_dir = Path(
        str(
            cfg[
                "SAVE_DIR"
            ]
        )
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_checkpoint_path = (
        save_dir
        / "last_model.pth"
    )

    final_checkpoint_path = (
        save_dir
        / "final_model.pth"
    )

    history_path = (
        save_dir
        / "training_history.csv"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset = FbankDataset(
        train_csv,
        cfg,
        training=True,
    )

    test_dataset = FbankDataset(
        test_csv,
        cfg,
        training=False,
    )

    train_loader = make_loader(
        train_dataset,
        cfg,
        device,
        shuffle=True,
    )

    test_loader = make_loader(
        test_dataset,
        cfg,
        device,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = DTFHybridModel(
        num_classes=4,

        stem_dim=int(
            cfg[
                "STEM_DIM"
            ]
        ),

        d_model=int(
            cfg[
                "D_MODEL"
            ]
        ),

        freq_patches=int(
            cfg[
                "FREQ_PATCHES"
            ]
        ),

        time_patches=int(
            cfg[
                "TIME_PATCHES"
            ]
        ),

        time_depth=int(
            cfg[
                "TIME_DEPTH"
            ]
        ),

        freq_depth=int(
            cfg[
                "FREQ_DEPTH"
            ]
        ),

        num_heads=int(
            cfg[
                "NHEAD"
            ]
        ),

        dropout=float(
            cfg[
                "DROPOUT"
            ]
        ),

        head_dropout=float(
            cfg[
                "HEAD_DROPOUT"
            ]
        ),

        d_state=int(
            cfg[
                "D_STATE"
            ]
        ),

        d_conv=int(
            cfg[
                "D_CONV"
            ]
        ),

        expand=int(
            cfg[
                "EXPAND"
            ]
        ),
    ).to(
        device
    )

    shape_test(
        train_loader,
        model,
        device,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"[Model] total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"[Model] trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss Weights
    # --------------------------------------------------------
    loss_weights = build_loss_weights(
        train_dataset.class_counts,
        cfg,
        device,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    head_parameters = (
        list(
            model.four_head.parameters()
        )
        + list(
            model.binary_head.parameters()
        )
        + list(
            model.abnormal_head.parameters()
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model
                    .frontend
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "FRONTEND_LR"
                    ]
                ),
            },

            {
                "params": (
                    model
                    .encoder
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "ENCODER_LR"
                    ]
                ),
            },

            {
                "params": (
                    head_parameters
                ),

                "lr": float(
                    cfg[
                        "HEAD_LR"
                    ]
                ),
            },
        ],

        weight_decay=float(
            cfg[
                "WEIGHT_DECAY"
            ]
        ),
    )

    base_learning_rates = [
        float(
            cfg[
                "FRONTEND_LR"
            ]
        ),

        float(
            cfg[
                "ENCODER_LR"
            ]
        ),

        float(
            cfg[
                "HEAD_LR"
            ]
        ),
    ]

    minimum_learning_rates = [
        float(
            cfg[
                "MIN_FRONTEND_LR"
            ]
        ),

        float(
            cfg[
                "MIN_ENCODER_LR"
            ]
        ),

        float(
            cfg[
                "MIN_HEAD_LR"
            ]
        ),
    ]

    use_amp = bool(
        cfg[
            "AMP"
        ]
        and device.type == "cuda"
    )

    scaler = make_scaler(
        use_amp
    )

    history = []

    print()

    print(
        "=" * 100
    )

    print(
        "D3: HIERARCHICAL MULTI-TASK TRAINING"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    for epoch in range(
        1,
        int(
            cfg[
                "EPOCHS"
            ]
        ) + 1,
    ):
        epoch_start_time = time.time()

        current_learning_rates = set_epoch_lrs(
            optimizer=optimizer,

            base_lrs=base_learning_rates,

            min_lrs=minimum_learning_rates,

            epoch=epoch,

            total_epochs=int(
                cfg[
                    "EPOCHS"
                ]
            ),

            warmup_epochs=int(
                cfg[
                    "WARMUP_EPOCHS"
                ]
            ),
        )

        train_result = train_one_epoch(
            loader=train_loader,

            model=model,

            optimizer=optimizer,

            device=device,

            scaler=scaler,

            use_amp=use_amp,

            loss_weights=loss_weights,

            cfg=cfg,
        )

        elapsed_time = (
            time.time()
            - epoch_start_time
        )

        stem_alpha = (
            model.get_dtf_alpha()
        )

        history_row = {
            "epoch": epoch,

            "total_loss": train_result[
                "total_loss"
            ],

            "four_loss": train_result[
                "four_loss"
            ],

            "binary_loss": train_result[
                "binary_loss"
            ],

            "abnormal_loss": train_result[
                "abnormal_loss"
            ],

            "frontend_lr": (
                current_learning_rates[
                    0
                ]
            ),

            "encoder_lr": (
                current_learning_rates[
                    1
                ]
            ),

            "head_lr": (
                current_learning_rates[
                    2
                ]
            ),

            "stem_alpha": (
                stem_alpha
            ),

            "seconds": (
                elapsed_time
            ),
        }

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state": (
                    model.state_dict()
                ),

                "optimizer_state": (
                    optimizer.state_dict()
                ),

                "config": deepcopy(
                    cfg
                ),

                "stem_alpha": (
                    stem_alpha
                ),

                "train_result": (
                    train_result
                ),
            },
            last_checkpoint_path,
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['EPOCHS']} | "
            f"Total "
            f"{train_result['total_loss']:.4f} | "
            f"Four "
            f"{train_result['four_loss']:.4f} | "
            f"Bin "
            f"{train_result['binary_loss']:.4f} | "
            f"Abn "
            f"{train_result['abnormal_loss']:.4f} | "
            f"Alpha "
            f"{stem_alpha:.4f} | "
            f"LR "
            f"{current_learning_rates[0]:.8f}/"
            f"{current_learning_rates[1]:.8f}/"
            f"{current_learning_rates[2]:.8f} | "
            f"{elapsed_time:.1f}s",
            flush=True,
        )

    # --------------------------------------------------------
    # Final Official Test
    # --------------------------------------------------------
    evaluation_result = evaluate(
        test_loader,
        model,
        device,
        cfg,
        use_amp,
    )

    print_final(
        evaluation_result
    )

    final_result = evaluation_result[
        "final"
    ]

    four_only_result = evaluation_result[
        "four_only"
    ]

    hierarchical_only_result = (
        evaluation_result[
            "hierarchical_only"
        ]
    )

    torch.save(
        {
            "epoch": int(
                cfg[
                    "EPOCHS"
                ]
            ),

            "model_state": (
                model.state_dict()
            ),

            "config": deepcopy(
                cfg
            ),

            "stem_alpha": (
                model.get_dtf_alpha()
            ),

            # ------------------------------------------------
            # Final Fused Prediction
            # ------------------------------------------------
            "final_score": final_result[
                "score"
            ],

            "final_sp": final_result[
                "sp"
            ],

            "final_se": final_result[
                "se"
            ],

            "final_accuracy": final_result[
                "accuracy"
            ],

            "final_macro_f1": final_result[
                "macro_f1"
            ],

            "final_recalls": final_result[
                "recalls"
            ].tolist(),

            "final_pred_counts": final_result[
                "pred_counts"
            ].tolist(),

            "final_four_cm": final_result[
                "four_cm"
            ].tolist(),

            "final_binary_cm": final_result[
                "binary_cm"
            ].tolist(),

            # ------------------------------------------------
            # Four-class Head Only
            # ------------------------------------------------
            "four_only_score": four_only_result[
                "score"
            ],

            "four_only_sp": four_only_result[
                "sp"
            ],

            "four_only_se": four_only_result[
                "se"
            ],

            "four_only_accuracy": four_only_result[
                "accuracy"
            ],

            "four_only_macro_f1": four_only_result[
                "macro_f1"
            ],

            "four_only_recalls": four_only_result[
                "recalls"
            ].tolist(),

            "four_only_pred_counts": four_only_result[
                "pred_counts"
            ].tolist(),

            "four_only_cm": four_only_result[
                "four_cm"
            ].tolist(),

            # ------------------------------------------------
            # Hierarchical Head Only
            # ------------------------------------------------
            "hierarchical_only_score": (
                hierarchical_only_result[
                    "score"
                ]
            ),

            "hierarchical_only_sp": (
                hierarchical_only_result[
                    "sp"
                ]
            ),

            "hierarchical_only_se": (
                hierarchical_only_result[
                    "se"
                ]
            ),

            "hierarchical_only_accuracy": (
                hierarchical_only_result[
                    "accuracy"
                ]
            ),

            "hierarchical_only_macro_f1": (
                hierarchical_only_result[
                    "macro_f1"
                ]
            ),

            "hierarchical_only_recalls": (
                hierarchical_only_result[
                    "recalls"
                ].tolist()
            ),

            "hierarchical_only_pred_counts": (
                hierarchical_only_result[
                    "pred_counts"
                ].tolist()
            ),

            "hierarchical_only_cm": (
                hierarchical_only_result[
                    "four_cm"
                ].tolist()
            ),

            # ------------------------------------------------
            # Auxiliary Heads
            # ------------------------------------------------
            "binary_head_accuracy": (
                evaluation_result[
                    "binary_head_accuracy"
                ]
            ),

            "binary_head_cm": (
                evaluation_result[
                    "binary_head_cm"
                ].tolist()
            ),

            "abnormal_head_accuracy": (
                evaluation_result[
                    "abnormal_head_accuracy"
                ]
            ),

            "abnormal_head_cm": (
                evaluation_result[
                    "abnormal_head_cm"
                ].tolist()
            ),
        },
        final_checkpoint_path,
    )

    print()

    print(
        "Last checkpoint:",
        last_checkpoint_path,
    )

    print(
        "Final checkpoint:",
        final_checkpoint_path,
    )

    print(
        "Training history:",
        history_path,
    )


if __name__ == "__main__":
    main()