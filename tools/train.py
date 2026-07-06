#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

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
PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

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
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_fbank"
    ),

    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_d4_dynamic_hierarchical_seed42"
    ),

    "EPOCHS": 50,

    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 4,

    "SEED": 42,

    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

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

    "BINARY_RESIDUAL_SCALE": 0.50,

    # 多任务损失
    "FOUR_LOSS_WEIGHT": 1.00,
    "BINARY_LOSS_WEIGHT": 0.25,
    "ABNORMAL_LOSS_WEIGHT": 0.75,

    # 类别权重
    "USE_FOUR_CLASS_WEIGHTS": False,
    "USE_BINARY_CLASS_WEIGHTS": False,
    "USE_ABNORMAL_CLASS_WEIGHTS": True,

    "ABNORMAL_MANUAL_WEIGHTS": [
        1.00,
        1.10,
        1.35,
    ],

    "FOUR_LABEL_SMOOTHING": 0.0,
    "BINARY_LABEL_SMOOTHING": 0.0,
    "ABNORMAL_LABEL_SMOOTHING": 0.0,

    # 动态层级融合
    "MIN_HIERARCHICAL_WEIGHT": 0.15,
    "MAX_HIERARCHICAL_WEIGHT": 0.60,

    # SpecAugment
    "USE_SPECAUGMENT": True,
    "TIME_MASK_MAX": 80,
    "FREQ_MASK_MAX": 16,

    "NUM_TIME_MASKS": 1,
    "NUM_FREQ_MASKS": 1,

    # 学习率
    "FRONTEND_LR": 3e-4,
    "ENCODER_LR": 1e-4,
    "HEAD_LR": 3e-4,

    "MIN_FRONTEND_LR": 3e-6,
    "MIN_ENCODER_LR": 1e-6,
    "MIN_HEAD_LR": 3e-6,

    "WARMUP_EPOCHS": 3,

    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

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
    optimizer,
    base_lrs,
    minimum_lrs,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
):
    if epoch <= warmup_epochs:
        warmup_ratio = (
            0.20
            + 0.80
            * epoch
            / max(
                warmup_epochs,
                1,
            )
        )

        current_lrs = [
            base_lr * warmup_ratio
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
            minimum_lr
            + (
                base_lr
                - minimum_lr
            )
            * cosine_ratio
            for base_lr, minimum_lr
            in zip(
                base_lrs,
                minimum_lrs,
            )
        ]

    for parameter_group, learning_rate in zip(
        optimizer.param_groups,
        current_lrs,
    ):
        parameter_group["lr"] = float(
            learning_rate
        )

    return current_lrs


# ============================================================
# SpecAugment
# ============================================================
def apply_specaugment(
    fbank: torch.Tensor,
    time_mask_max: int,
    frequency_mask_max: int,
    num_time_masks: int = 1,
    num_frequency_masks: int = 1,
) -> torch.Tensor:
    if fbank.ndim != 2:
        raise ValueError(
            "SpecAugment输入必须为[T,F]。"
        )

    x = fbank.clone()

    time_frames = int(
        x.shape[0]
    )

    frequency_bins = int(
        x.shape[1]
    )

    mask_value = x.mean()

    for _ in range(
        max(num_time_masks, 0)
    ):
        if time_mask_max <= 0:
            break

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

    for _ in range(
        max(num_frequency_masks, 0)
    ):
        if frequency_mask_max <= 0:
            break

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
# Dataset
# ============================================================
class FbankDataset(Dataset):
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
                cfg["FBANK_FRAMES"]
            ),
            int(
                cfg["FBANK_MELS"]
            ),
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(
                self.csv_path
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
                f"CSV缺少列："
                f"{sorted(missing_columns)}"
            )

        self.dataframe["label"] = (
            self.dataframe["label"]
            .astype(int)
        )

        self.labels = self.dataframe[
            "label"
        ].to_numpy(
            dtype=np.int64
        )

        if np.any(
            (self.labels < 0)
            | (self.labels > 3)
        ):
            raise ValueError(
                "标签必须位于0到3。"
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
            str(raw_path)
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
            row["fbank_path"]
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
                f"Fbank包含NaN或Inf："
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
                fbank=x,

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

                num_time_masks=int(
                    self.cfg[
                        "NUM_TIME_MASKS"
                    ]
                ),

                num_frequency_masks=int(
                    self.cfg[
                        "NUM_FREQ_MASKS"
                    ]
                ),
            )

        x = x.unsqueeze(0)

        y = torch.tensor(
            int(row["label"]),
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
        cfg["NUM_WORKERS"]
    )

    arguments = {
        "dataset": dataset,

        "batch_size": int(
            cfg["BATCH_SIZE"]
        ),

        "shuffle": shuffle,

        "num_workers": workers,

        "pin_memory": (
            device.type == "cuda"
        ),

        "persistent_workers": (
            workers > 0
        ),

        "drop_last": False,
    }

    if workers > 0:
        arguments[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **arguments
    )


# ============================================================
# Loss Weights
# ============================================================
def build_loss_weights(
    cfg,
    device,
):
    four_weight = None
    binary_weight = None

    if bool(
        cfg[
            "USE_ABNORMAL_CLASS_WEIGHTS"
        ]
    ):
        abnormal_weight = torch.tensor(
            cfg[
                "ABNORMAL_MANUAL_WEIGHTS"
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        abnormal_weight = None

    print(
        "[Loss] Four-class weight:",
        four_weight,
    )

    print(
        "[Loss] Binary weight:",
        binary_weight,
    )

    print(
        "[Loss] Abnormal weight:",
        (
            None
            if abnormal_weight is None
            else abnormal_weight.detach()
            .cpu()
            .tolist()
        ),
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
    four_loss = F.cross_entropy(
        outputs["four_logits"],
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

    binary_labels = (
        labels > 0
    ).long()

    binary_loss = F.cross_entropy(
        outputs["binary_logits"],
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

    total_batches = len(loader)

    accumulation_steps = int(
        cfg["ACCUM_STEPS"]
    )

    total_samples = 0
    total_abnormal_samples = 0

    accumulated_total_loss = 0.0
    accumulated_four_loss = 0.0
    accumulated_binary_loss = 0.0
    accumulated_abnormal_loss = 0.0

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    epoch_start_time = time.time()

    for batch_index, (
        x,
        y,
    ) in enumerate(loader):
        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        completed_batches = (
            batch_index + 1
        )

        accumulation_window_start = (
            batch_index
            // accumulation_steps
        ) * accumulation_steps

        accumulation_window_end = min(
            accumulation_window_start
            + accumulation_steps,
            total_batches,
        )

        current_accumulation_size = max(
            accumulation_window_end
            - accumulation_window_start,
            1,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(x)

            loss_result = (
                calculate_multitask_loss(
                    outputs=outputs,
                    labels=y,
                    loss_weights=loss_weights,
                    cfg=cfg,
                )
            )

            backward_loss = (
                loss_result[
                    "total_loss"
                ]
                / current_accumulation_size
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

        total_abnormal_samples += (
            abnormal_count
        )

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

        should_update = (
            completed_batches
            == accumulation_window_end
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
            % int(
                cfg[
                    "PRINT_INTERVAL"
                ]
            )
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
                f"{remaining_seconds / 60:.1f}min",
                flush=True,
            )

    return {
        "total_loss": (
            accumulated_total_loss
            / max(
                total_samples,
                1,
            )
        ),

        "four_loss": (
            accumulated_four_loss
            / max(
                total_samples,
                1,
            )
        ),

        "binary_loss": (
            accumulated_binary_loss
            / max(
                total_samples,
                1,
            )
        ),

        "abnormal_loss": (
            accumulated_abnormal_loss
            / max(
                total_abnormal_samples,
                1,
            )
        ),
    }


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    four_cm = confusion_matrix(
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
            four_cm[0].sum()
        ),
        1,
    )

    abnormal_total = max(
        int(
            four_cm[1:].sum()
        ),
        1,
    )

    specificity = (
        100.0
        * float(
            four_cm[0, 0]
        )
        / normal_total
    )

    sensitivity = (
        100.0
        * float(
            four_cm[1, 1]
            + four_cm[2, 2]
            + four_cm[3, 3]
        )
        / abnormal_total
    )

    score = (
        specificity
        + sensitivity
    ) / 2.0

    binary_true = (
        y_true > 0
    ).astype(np.int64)

    binary_pred = (
        y_pred > 0
    ).astype(np.int64)

    return {
        "score": float(score),

        "sp": float(specificity),

        "se": float(sensitivity),

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

        "recalls": recall_score(
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
        ),

        "pred_counts": np.bincount(
            y_pred,
            minlength=4,
        ),

        "four_cm": four_cm,

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

    all_binary_predictions = []

    all_abnormal_true = []
    all_abnormal_predictions = []

    binary_confidence_values = []
    hierarchical_weight_values = []

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
            outputs = model(x)

            probabilities = (
                model.build_probabilities(
                    outputs=outputs,

                    four_weight=None,

                    minimum_hierarchical_weight=float(
                        cfg[
                            "MIN_HIERARCHICAL_WEIGHT"
                        ]
                    ),

                    maximum_hierarchical_weight=float(
                        cfg[
                            "MAX_HIERARCHICAL_WEIGHT"
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

        hierarchical_predictions = torch.argmax(
            probabilities[
                "hierarchical_probability"
            ],
            dim=1,
        )

        binary_predictions = torch.argmax(
            probabilities[
                "binary_probability"
            ],
            dim=1,
        )

        abnormal_predictions = torch.argmax(
            probabilities[
                "abnormal_probability"
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

        all_binary_predictions.append(
            binary_predictions.cpu()
        )

        binary_confidence_values.append(
            probabilities[
                "binary_confidence"
            ].detach().cpu()
        )

        hierarchical_weight_values.append(
            probabilities[
                "hierarchical_weight"
            ].detach().cpu()
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

            all_abnormal_predictions.append(
                abnormal_predictions[
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

    binary_pred = torch.cat(
        all_binary_predictions
    ).numpy()

    binary_true = (
        y_true > 0
    ).astype(np.int64)

    binary_confidence = torch.cat(
        binary_confidence_values
    ).numpy()

    hierarchical_weights = torch.cat(
        hierarchical_weight_values
    ).numpy()

    binary_head_cm = confusion_matrix(
        binary_true,
        binary_pred,
        labels=[
            0,
            1,
        ],
    )

    binary_head_accuracy = (
        accuracy_score(
            binary_true,
            binary_pred,
        )
        * 100.0
    )

    abnormal_true = torch.cat(
        all_abnormal_true
    ).numpy()

    abnormal_pred = torch.cat(
        all_abnormal_predictions
    ).numpy()

    abnormal_head_cm = confusion_matrix(
        abnormal_true,
        abnormal_pred,
        labels=[
            0,
            1,
            2,
        ],
    )

    abnormal_head_accuracy = (
        accuracy_score(
            abnormal_true,
            abnormal_pred,
        )
        * 100.0
    )

    abnormal_recalls = recall_score(
        abnormal_true,
        abnormal_pred,
        labels=[
            0,
            1,
            2,
        ],
        average=None,
        zero_division=0,
    )

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

        "binary_head_accuracy": float(
            binary_head_accuracy
        ),

        "binary_head_cm": (
            binary_head_cm
        ),

        "abnormal_head_accuracy": float(
            abnormal_head_accuracy
        ),

        "abnormal_head_cm": (
            abnormal_head_cm
        ),

        "abnormal_head_recalls": (
            abnormal_recalls
        ),

        "mean_binary_confidence": float(
            binary_confidence.mean()
        ),

        "mean_hierarchical_weight": float(
            hierarchical_weights.mean()
        ),

        "min_hierarchical_weight": float(
            hierarchical_weights.min()
        ),

        "max_hierarchical_weight": float(
            hierarchical_weights.max()
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
    cfg,
):
    model.eval()

    x, _ = next(
        iter(loader)
    )

    x = x[:2].to(device)

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

    outputs = model(x)

    probabilities = (
        model.build_probabilities(
            outputs=outputs,

            four_weight=None,

            minimum_hierarchical_weight=float(
                cfg[
                    "MIN_HIERARCHICAL_WEIGHT"
                ]
            ),

            maximum_hierarchical_weight=float(
                cfg[
                    "MAX_HIERARCHICAL_WEIGHT"
                ]
            ),
        )
    )

    print(
        "[Shape] Fbank:",
        tuple(x.shape),
    )

    print(
        "[Shape] Stem:",
        tuple(stem_map.shape),
    )

    print(
        "[Shape] Stage1:",
        tuple(stage1_map.shape),
    )

    print(
        "[Shape] Stage2:",
        tuple(stage2_map.shape),
    )

    print(
        "[Shape] Patch:",
        tuple(patch_map.shape),
    )

    print(
        "[Shape] Tokens:",
        tuple(tokens.shape),
    )

    print(
        "[Shape] Four/Binary/Abnormal:",
        tuple(
            outputs[
                "four_logits"
            ].shape
        ),
        tuple(
            outputs[
                "binary_logits"
            ].shape
        ),
        tuple(
            outputs[
                "abnormal_logits"
            ].shape
        ),
    )

    print(
        "[Shape] Final probability:",
        tuple(
            probabilities[
                "final_probability"
            ].shape
        ),
    )

    assert tuple(
        patch_map.shape[1:]
    ) == (
        256,
        100,
        16,
    )

    assert tuple(
        tokens.shape[1:]
    ) == (
        1600,
        256,
    )

    assert tuple(
        outputs[
            "four_logits"
        ].shape[1:]
    ) == (
        4,
    )

    assert tuple(
        outputs[
            "binary_logits"
        ].shape[1:]
    ) == (
        2,
    )

    assert tuple(
        outputs[
            "abnormal_logits"
        ].shape[1:]
    ) == (
        3,
    )

    print(
        "[PASS] D4模型连接成功。",
        flush=True,
    )


# ============================================================
# Print
# ============================================================
def print_metric_block(
    title,
    result,
):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)

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
        result["four_cm"]
    )

    print()
    print(
        "Binary confusion matrix:"
    )

    print(
        result["binary_cm"]
    )


def print_final(
    evaluation,
):
    print()
    print("=" * 80)
    print(
        "FINAL OFFICIAL TEST RESULT"
    )
    print("=" * 80)

    print_metric_block(
        "FINAL DYNAMIC FUSED PREDICTION",
        evaluation["final"],
    )

    print_metric_block(
        "FOUR-CLASS HEAD ONLY",
        evaluation["four_only"],
    )

    print_metric_block(
        "HIERARCHICAL HEAD ONLY",
        evaluation[
            "hierarchical_only"
        ],
    )

    print()
    print("-" * 80)
    print(
        "AUXILIARY HEAD RESULTS"
    )
    print("-" * 80)

    print(
        f"Binary Head Accuracy: "
        f"{evaluation['binary_head_accuracy']:.4f}"
    )

    print(
        "Binary Head Confusion Matrix:"
    )

    print(
        evaluation[
            "binary_head_cm"
        ]
    )

    print()

    print(
        f"Abnormal Head Accuracy: "
        f"{evaluation['abnormal_head_accuracy']:.4f}"
    )

    print(
        "Abnormal Head Recall "
        "[Crackle, Wheeze, Both]:",
        np.round(
            evaluation[
                "abnormal_head_recalls"
            ],
            4,
        ).tolist(),
    )

    print(
        "Abnormal Head Confusion Matrix:"
    )

    print(
        evaluation[
            "abnormal_head_cm"
        ]
    )

    print()

    print(
        f"Mean Binary Confidence: "
        f"{evaluation['mean_binary_confidence']:.4f}"
    )

    print(
        f"Mean Hierarchical Weight: "
        f"{evaluation['mean_hierarchical_weight']:.4f}"
    )

    print(
        f"Hierarchical Weight Range: "
        f"{evaluation['min_hierarchical_weight']:.4f}"
        f" - "
        f"{evaluation['max_hierarchical_weight']:.4f}"
    )


# ============================================================
# Main
# ============================================================
def main():
    cfg = CONFIG

    set_seed(
        int(cfg["SEED"])
    )

    device = torch.device(
        "cuda"
        if (
            str(cfg["DEVICE"]) == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    use_amp = bool(
        cfg["AMP"]
        and device.type == "cuda"
    )

    print(
        "[INFO] device:",
        device,
    )

    if device.type == "cuda":
        print(
            "[INFO] GPU:",
            torch.cuda.get_device_name(0),
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
            "mamba_ssm导入失败。"
        )

    root = Path(
        str(cfg["ROOT"])
    )

    train_csv = (
        root
        / "train_index.csv"
    )

    test_csv = (
        root
        / "test_index.csv"
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
        "[Experiment] D4："
        "DTF Stem"
        " + Progressive Downsampling"
        " + Time-Mamba"
        " + Frequency-Attention"
        " + Consistent Binary Residual"
        " + Dynamic Hierarchical Fusion"
    )

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

    model = DTFHybridModel(
        num_classes=4,

        stem_dim=int(
            cfg["STEM_DIM"]
        ),

        d_model=int(
            cfg["D_MODEL"]
        ),

        freq_patches=int(
            cfg["FREQ_PATCHES"]
        ),

        time_patches=int(
            cfg["TIME_PATCHES"]
        ),

        time_depth=int(
            cfg["TIME_DEPTH"]
        ),

        freq_depth=int(
            cfg["FREQ_DEPTH"]
        ),

        num_heads=int(
            cfg["NHEAD"]
        ),

        dropout=float(
            cfg["DROPOUT"]
        ),

        head_dropout=float(
            cfg["HEAD_DROPOUT"]
        ),

        d_state=int(
            cfg["D_STATE"]
        ),

        d_conv=int(
            cfg["D_CONV"]
        ),

        expand=int(
            cfg["EXPAND"]
        ),

        binary_residual_scale=float(
            cfg[
                "BINARY_RESIDUAL_SCALE"
            ]
        ),
    ).to(device)

    shape_test(
        train_loader,
        model,
        device,
        cfg,
    )

    loss_weights = build_loss_weights(
        cfg,
        device,
    )

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
                    model.frontend.parameters()
                ),

                "lr": float(
                    cfg["FRONTEND_LR"]
                ),
            },

            {
                "params": (
                    model.encoder.parameters()
                ),

                "lr": float(
                    cfg["ENCODER_LR"]
                ),
            },

            {
                "params": head_parameters,

                "lr": float(
                    cfg["HEAD_LR"]
                ),
            },
        ],

        weight_decay=float(
            cfg["WEIGHT_DECAY"]
        ),
    )

    base_lrs = [
        float(cfg["FRONTEND_LR"]),
        float(cfg["ENCODER_LR"]),
        float(cfg["HEAD_LR"]),
    ]

    minimum_lrs = [
        float(cfg["MIN_FRONTEND_LR"]),
        float(cfg["MIN_ENCODER_LR"]),
        float(cfg["MIN_HEAD_LR"]),
    ]

    scaler = make_scaler(
        use_amp
    )

    save_directory = Path(
        str(cfg["SAVE_DIR"])
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        save_directory
        / "training_history.csv"
    )

    final_model_path = (
        save_directory
        / "final_model.pth"
    )

    history = []

    print()
    print("=" * 100)
    print(
        "D4: CONSISTENT DYNAMIC "
        "HIERARCHICAL TRAINING"
    )
    print("=" * 100)

    for epoch in range(
        1,
        int(cfg["EPOCHS"]) + 1,
    ):
        current_lrs = set_epoch_lrs(
            optimizer=optimizer,

            base_lrs=base_lrs,

            minimum_lrs=minimum_lrs,

            epoch=epoch,

            total_epochs=int(
                cfg["EPOCHS"]
            ),

            warmup_epochs=int(
                cfg["WARMUP_EPOCHS"]
            ),
        )

        epoch_start = time.time()

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

        elapsed = (
            time.time()
            - epoch_start
        )

        history.append(
            {
                "epoch": epoch,

                "total_loss": (
                    train_result[
                        "total_loss"
                    ]
                ),

                "four_loss": (
                    train_result[
                        "four_loss"
                    ]
                ),

                "binary_loss": (
                    train_result[
                        "binary_loss"
                    ]
                ),

                "abnormal_loss": (
                    train_result[
                        "abnormal_loss"
                    ]
                ),

                "frontend_lr": (
                    current_lrs[0]
                ),

                "encoder_lr": (
                    current_lrs[1]
                ),

                "head_lr": (
                    current_lrs[2]
                ),

                "dtf_alpha": (
                    model.get_dtf_alpha()
                ),

                "seconds": elapsed,
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
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
            f"{model.get_dtf_alpha():.4f} | "

            f"LR "
            f"{current_lrs[0]:.8f}/"
            f"{current_lrs[1]:.8f}/"
            f"{current_lrs[2]:.8f} | "

            f"{elapsed:.1f}s",
            flush=True,
        )

    evaluation = evaluate(
        loader=test_loader,

        model=model,

        device=device,

        cfg=cfg,

        use_amp=use_amp,
    )

    print_final(
        evaluation
    )

    final_result = evaluation[
        "final"
    ]

    torch.save(
        {
            "epoch": int(
                cfg["EPOCHS"]
            ),

            "model_state": (
                model.state_dict()
            ),

            "config": deepcopy(
                cfg
            ),

            "score": (
                final_result["score"]
            ),

            "sp": (
                final_result["sp"]
            ),

            "se": (
                final_result["se"]
            ),

            "accuracy": (
                final_result[
                    "accuracy"
                ]
            ),

            "macro_f1": (
                final_result[
                    "macro_f1"
                ]
            ),

            "recalls": (
                final_result[
                    "recalls"
                ].tolist()
            ),

            "pred_counts": (
                final_result[
                    "pred_counts"
                ].tolist()
            ),

            "four_cm": (
                final_result[
                    "four_cm"
                ].tolist()
            ),

            "binary_cm": (
                final_result[
                    "binary_cm"
                ].tolist()
            ),

            "binary_head_accuracy": (
                evaluation[
                    "binary_head_accuracy"
                ]
            ),

            "binary_head_cm": (
                evaluation[
                    "binary_head_cm"
                ].tolist()
            ),

            "abnormal_head_accuracy": (
                evaluation[
                    "abnormal_head_accuracy"
                ]
            ),

            "abnormal_head_recalls": (
                evaluation[
                    "abnormal_head_recalls"
                ].tolist()
            ),

            "abnormal_head_cm": (
                evaluation[
                    "abnormal_head_cm"
                ].tolist()
            ),
        },
        final_model_path,
    )

    print()
    print(
        "Training history:",
        history_path,
    )

    print(
        "Final model:",
        final_model_path,
    )


if __name__ == "__main__":
    main()