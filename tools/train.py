#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. 项目路径与模型导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import HAS_MAMBA, DTFHybridModel


# ============================================================
# 2. 配置
# ============================================================
CONFIG: Dict[str, object] = {
    # --------------------------------------------------------
    # 数据路径
    # --------------------------------------------------------
    "ROOT": "/data/dingcong/hybrid/icbhi_official_fbank",

    # --------------------------------------------------------
    # 保存路径
    # --------------------------------------------------------
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_d6_soft_hierarchical_seed42"
    ),

    # --------------------------------------------------------
    # 训练配置
    # --------------------------------------------------------
    "EPOCHS": 50,
    "VAL_RATIO": 0.20,

    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 4,

    "SEED": 42,

    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # 前几轮不参与最佳模型选择，避免随机初始化阶段被误选
    "BEST_EPOCH_START": 5,

    # --------------------------------------------------------
    # Fbank输入尺寸
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
    "ADAPTER_DROPOUT": 0.15,
    "ADAPTER_BOTTLENECK_RATIO": 0.50,

    # --------------------------------------------------------
    # 软层级融合
    #
    # Binary Head不确定：
    # 层级分支权重接近0.10
    #
    # Binary Head确定：
    # 层级分支权重最高0.35
    # --------------------------------------------------------
    "MIN_HIERARCHICAL_WEIGHT": 0.10,
    "MAX_HIERARCHICAL_WEIGHT": 0.35,

    # --------------------------------------------------------
    # 多任务损失
    #
    # Total Loss =
    # 1.00 * Four Loss
    # + 0.30 * Binary Loss
    # + 0.50 * Abnormal Loss
    # + 0.05 * Consistency Loss
    # --------------------------------------------------------
    "FOUR_LOSS_WEIGHT": 1.00,
    "BINARY_LOSS_WEIGHT": 0.30,
    "ABNORMAL_LOSS_WEIGHT": 0.50,
    "CONSISTENCY_LOSS_WEIGHT": 0.05,

    # --------------------------------------------------------
    # 类别权重
    #
    # Four Head不使用类别权重，避免再次过度预测少数类
    #
    # Abnormal Head使用轻度权重：
    # Crackle / Wheeze / Both
    # --------------------------------------------------------
    "ABNORMAL_MANUAL_WEIGHTS": [
        1.00,
        1.10,
        1.25,
    ],

    # --------------------------------------------------------
    # Label Smoothing
    # --------------------------------------------------------
    "FOUR_LABEL_SMOOTHING": 0.00,
    "BINARY_LABEL_SMOOTHING": 0.00,
    "ABNORMAL_LABEL_SMOOTHING": 0.00,

    # --------------------------------------------------------
    # SpecAugment
    # --------------------------------------------------------
    "USE_SPECAUGMENT": True,

    "TIME_MASK_MAX": 80,
    "FREQ_MASK_MAX": 16,

    "NUM_TIME_MASKS": 1,
    "NUM_FREQ_MASKS": 1,

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
    # 最佳模型选择
    #
    # Selection =
    # 0.70 * ICBHI Score
    # + 0.30 * Macro-F1 * 100
    #
    # 不再用Accuracy选择，防止大量预测Normal。
    # --------------------------------------------------------
    "SELECTION_SCORE_WEIGHT": 0.70,
    "SELECTION_MACRO_F1_WEIGHT": 0.30,

    # --------------------------------------------------------
    # 日志
    # --------------------------------------------------------
    "PRINT_INTERVAL": 50,
}


# ============================================================
# 3. 随机种子
# ============================================================
def set_seed(seed: int) -> None:
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
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


# ============================================================
# 4. AMP GradScaler
# ============================================================
def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


# ============================================================
# 5. 保存CPU State Dict
# ============================================================
def state_dict_to_cpu(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# ============================================================
# 6. 患者ID
# ============================================================
def infer_patient_id(
    path_value: str,
) -> str:
    """
    ICBHI文件通常为：

        101_1b1_Al_sc_Meditron_xxx.npy

    第一个下划线前面的101为患者ID。
    """

    filename_stem = Path(
        str(path_value)
    ).stem

    return filename_stem.split("_")[0]


def add_patient_groups(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    candidate_columns = [
        "patient_id",
        "patient",
        "patient_number",
        "subject_id",
        "subject",
    ]

    for column in candidate_columns:
        if column in dataframe.columns:
            dataframe["_patient_group"] = (
                dataframe[column]
                .astype(str)
            )

            return dataframe

    dataframe["_patient_group"] = (
        dataframe["fbank_path"]
        .astype(str)
        .map(infer_patient_id)
    )

    return dataframe


# ============================================================
# 7. 患者级内部训练/验证划分
# ============================================================
def patient_level_split(
    dataframe: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = add_patient_groups(
        dataframe
    )

    labels = dataframe[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    groups = dataframe[
        "_patient_group"
    ].to_numpy()

    if len(np.unique(groups)) < 2:
        raise RuntimeError(
            "患者数量不足，无法进行患者级划分。"
        )

    full_distribution = np.bincount(
        labels,
        minlength=4,
    ).astype(np.float64)

    full_distribution = (
        full_distribution
        / full_distribution.sum()
    )

    best_candidate = None
    best_difference = float("inf")

    # 搜索较均衡的患者级划分
    for offset in range(500):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed + offset,
        )

        train_indices, validation_indices = next(
            splitter.split(
                dataframe,
                labels,
                groups,
            )
        )

        train_dataframe = dataframe.iloc[
            train_indices
        ].reset_index(drop=True)

        validation_dataframe = dataframe.iloc[
            validation_indices
        ].reset_index(drop=True)

        train_counts = np.bincount(
            train_dataframe["label"],
            minlength=4,
        )

        validation_counts = np.bincount(
            validation_dataframe["label"],
            minlength=4,
        )

        # 训练集和验证集必须均包含四类
        if np.any(train_counts == 0):
            continue

        if np.any(validation_counts == 0):
            continue

        train_patients = set(
            train_dataframe["_patient_group"]
        )

        validation_patients = set(
            validation_dataframe["_patient_group"]
        )

        if train_patients & validation_patients:
            continue

        validation_distribution = (
            validation_counts.astype(np.float64)
            / validation_counts.sum()
        )

        distribution_difference = float(
            np.abs(
                validation_distribution
                - full_distribution
            ).sum()
        )

        size_difference = abs(
            len(validation_dataframe)
            / len(dataframe)
            - val_ratio
        )

        candidate_difference = (
            distribution_difference
            + size_difference
        )

        if candidate_difference < best_difference:
            best_difference = candidate_difference

            best_candidate = (
                train_dataframe,
                validation_dataframe,
            )

    if best_candidate is None:
        raise RuntimeError(
            "无法获得同时包含四类的患者级内部验证集。"
        )

    return best_candidate


# ============================================================
# 8. Warmup + Cosine学习率
# ============================================================
def set_epoch_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs,
    minimum_lrs,
    epoch: int,
    schedule_total_epochs: int,
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
            schedule_total_epochs
            - warmup_epochs,
            1,
        )

        cosine_step = min(
            epoch - warmup_epochs,
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
# 9. SpecAugment
# ============================================================
def apply_specaugment(
    fbank: torch.Tensor,
    time_mask_max: int,
    frequency_mask_max: int,
    num_time_masks: int = 1,
    num_frequency_masks: int = 1,
) -> torch.Tensor:
    """
    输入：
        [T,F]

    输出：
        [T,F]
    """

    if fbank.ndim != 2:
        raise ValueError(
            "SpecAugment输入必须为[T,F]，"
            f"当前为{tuple(fbank.shape)}。"
        )

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
    for _ in range(
        max(
            int(num_time_masks),
            0,
        )
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
                time_frames - time_width,
            )

            x[
                time_start:
                time_start + time_width,
                :
            ] = mask_value

    # --------------------------------------------------------
    # Frequency Mask
    # --------------------------------------------------------
    for _ in range(
        max(
            int(num_frequency_masks),
            0,
        )
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
# 10. Dataset
# ============================================================
class FbankDataset(Dataset):
    """
    DataFrame必须包含：

        fbank_path
        label

    单个Fbank：
        [798,128]

    返回：
        x: [1,798,128]
        y: Long Tensor
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        base_directory: Path,
        cfg,
        training: bool,
    ) -> None:
        super().__init__()

        self.dataframe = dataframe.reset_index(
            drop=True
        ).copy()

        self.base_directory = Path(
            base_directory
        )

        self.cfg = cfg
        self.training = training

        self.expected_shape = (
            int(cfg["FBANK_FRAMES"]),
            int(cfg["FBANK_MELS"]),
        )

        required_columns = {
            "fbank_path",
            "label",
        }

        missing_columns = (
            required_columns
            - set(self.dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "数据表缺少列："
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
                "标签必须位于0到3之间。"
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
            f"training={self.training}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def resolve_path(
        self,
        raw_path,
    ) -> Path:
        fbank_path = Path(
            str(raw_path)
        )

        if fbank_path.exists():
            return fbank_path

        candidate_path = (
            self.base_directory
            / fbank_path
        )

        if candidate_path.exists():
            return candidate_path

        raise FileNotFoundError(
            f"Fbank文件不存在：{raw_path}"
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

        if tuple(fbank.shape) != self.expected_shape:
            raise ValueError(
                f"Fbank尺寸错误：{fbank_path}\n"
                f"当前={tuple(fbank.shape)}，"
                f"要求={self.expected_shape}"
            )

        if not np.isfinite(fbank).all():
            raise ValueError(
                f"Fbank包含NaN或Inf：{fbank_path}"
            )

        x = torch.from_numpy(
            fbank
        ).float()

        if (
            self.training
            and bool(
                self.cfg["USE_SPECAUGMENT"]
            )
        ):
            x = apply_specaugment(
                fbank=x,

                time_mask_max=int(
                    self.cfg["TIME_MASK_MAX"]
                ),

                frequency_mask_max=int(
                    self.cfg["FREQ_MASK_MAX"]
                ),

                num_time_masks=int(
                    self.cfg["NUM_TIME_MASKS"]
                ),

                num_frequency_masks=int(
                    self.cfg["NUM_FREQ_MASKS"]
                ),
            )

        # [T,F] -> [1,T,F]
        x = x.unsqueeze(0)

        y = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# 11. DataLoader
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

    loader_arguments = {
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
        loader_arguments[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **loader_arguments
    )


# ============================================================
# 12. 构建模型
# ============================================================
def build_model(
    cfg,
    device,
):
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

        adapter_dropout=float(
            cfg["ADAPTER_DROPOUT"]
        ),

        adapter_bottleneck_ratio=float(
            cfg[
                "ADAPTER_BOTTLENECK_RATIO"
            ]
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
    ).to(device)

    return model


# ============================================================
# 13. 优化器
# ============================================================
def build_optimizer(
    model,
    cfg,
):
    head_parameters = (
        list(
            model.four_adapter.parameters()
        )
        + list(
            model.binary_adapter.parameters()
        )
        + list(
            model.abnormal_adapter.parameters()
        )
        + list(
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

    base_learning_rates = [
        float(cfg["FRONTEND_LR"]),
        float(cfg["ENCODER_LR"]),
        float(cfg["HEAD_LR"]),
    ]

    minimum_learning_rates = [
        float(cfg["MIN_FRONTEND_LR"]),
        float(cfg["MIN_ENCODER_LR"]),
        float(cfg["MIN_HEAD_LR"]),
    ]

    return (
        optimizer,
        base_learning_rates,
        minimum_learning_rates,
    )


# ============================================================
# 14. 损失权重
# ============================================================
def build_loss_weights(
    cfg,
    device,
):
    abnormal_weight = torch.tensor(
        cfg["ABNORMAL_MANUAL_WEIGHTS"],
        dtype=torch.float32,
        device=device,
    )

    if abnormal_weight.numel() != 3:
        raise ValueError(
            "ABNORMAL_MANUAL_WEIGHTS"
            "必须包含3个值。"
        )

    print(
        "[Loss] Four-class weight: None",
        flush=True,
    )

    print(
        "[Loss] Binary weight: None",
        flush=True,
    )

    print(
        "[Loss] Abnormal weight:",
        abnormal_weight
        .detach()
        .cpu()
        .tolist(),
        flush=True,
    )

    return {
        "four": None,
        "binary": None,
        "abnormal": abnormal_weight,
    }


# ============================================================
# 15. 多任务损失
# ============================================================
def calculate_multitask_loss(
    outputs,
    labels,
    model,
    loss_weights,
    cfg,
):
    # --------------------------------------------------------
    # Four-class Loss
    # --------------------------------------------------------
    four_loss = F.cross_entropy(
        outputs["four_logits"],
        labels,

        weight=loss_weights["four"],

        label_smoothing=float(
            cfg["FOUR_LABEL_SMOOTHING"]
        ),
    )

    # --------------------------------------------------------
    # Binary Cross Entropy
    #
    # 不再使用Focal Loss
    # --------------------------------------------------------
    binary_labels = (
        labels > 0
    ).long()

    binary_loss = F.cross_entropy(
        outputs["binary_logits"],
        binary_labels,

        weight=loss_weights["binary"],

        label_smoothing=float(
            cfg["BINARY_LABEL_SMOOTHING"]
        ),
    )

    # --------------------------------------------------------
    # Abnormal Subtype Loss
    #
    # 原始标签：
    # 1=Crackle
    # 2=Wheeze
    # 3=Both
    #
    # 异常头：
    # 0=Crackle
    # 1=Wheeze
    # 2=Both
    # --------------------------------------------------------
    abnormal_mask = (
        labels > 0
    )

    abnormal_count = int(
        abnormal_mask.sum().item()
    )

    if abnormal_count > 0:
        abnormal_labels = (
            labels[abnormal_mask]
            - 1
        )

        abnormal_logits = outputs[
            "abnormal_logits"
        ][abnormal_mask]

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

    # --------------------------------------------------------
    # Binary/Four一致性损失
    #
    # detach Four Head，避免Binary Loss干扰Four Head。
    # --------------------------------------------------------
    probabilities = model.build_probabilities(
        outputs=outputs,

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

    consistency_loss = F.mse_loss(
        probabilities[
            "binary_probability"
        ],

        probabilities[
            "four_binary_probability"
        ].detach(),
    )

    total_loss = (
        float(
            cfg["FOUR_LOSS_WEIGHT"]
        )
        * four_loss

        + float(
            cfg["BINARY_LOSS_WEIGHT"]
        )
        * binary_loss

        + float(
            cfg["ABNORMAL_LOSS_WEIGHT"]
        )
        * abnormal_loss

        + float(
            cfg[
                "CONSISTENCY_LOSS_WEIGHT"
            ]
        )
        * consistency_loss
    )

    return {
        "total_loss": total_loss,

        "four_loss": four_loss,

        "binary_loss": binary_loss,

        "abnormal_loss": abnormal_loss,

        "consistency_loss": (
            consistency_loss
        ),

        "abnormal_count": (
            abnormal_count
        ),
    }


# ============================================================
# 16. 单轮训练
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
    accumulated_consistency_loss = 0.0

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    epoch_start_time = time.time()

    for batch_index, (x, y) in enumerate(loader):
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

            loss_result = calculate_multitask_loss(
                outputs=outputs,
                labels=y,
                model=model,
                loss_weights=loss_weights,
                cfg=cfg,
            )

            backward_loss = (
                loss_result["total_loss"]
                / current_accumulation_size
            )

        scaler.scale(
            backward_loss
        ).backward()

        batch_size = int(
            y.shape[0]
        )

        abnormal_count = int(
            loss_result["abnormal_count"]
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

        accumulated_consistency_loss += (
            float(
                loss_result[
                    "consistency_loss"
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
                    cfg["GRAD_CLIP"]
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
                cfg["PRINT_INTERVAL"]
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

            if device.type == "cuda":
                allocated_memory = (
                    torch.cuda.memory_allocated(
                        device
                    )
                    / 1024**3
                )

                reserved_memory = (
                    torch.cuda.memory_reserved(
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

                f"Cons "
                f"{loss_result['consistency_loss'].item():.4f} | "

                f"ETA "
                f"{remaining_seconds / 60:.1f}min | "

                f"GPU "
                f"{allocated_memory:.2f}/"
                f"{reserved_memory:.2f}GB",
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

        "consistency_loss": (
            accumulated_consistency_loss
            / max(
                total_samples,
                1,
            )
        ),
    }


# ============================================================
# 17. 收集概率
# ============================================================
@torch.no_grad()
def collect_probabilities(
    loader,
    model,
    device,
    use_amp: bool,
    cfg,
):
    model.eval()

    all_labels = []

    all_final_probabilities = []
    all_four_probabilities = []
    all_binary_probabilities = []
    all_abnormal_probabilities = []
    all_hierarchical_probabilities = []

    all_binary_confidences = []
    all_hierarchical_weights = []

    for x, y in loader:
        x = x.to(
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

        all_labels.append(
            y.cpu()
        )

        all_final_probabilities.append(
            probabilities[
                "final_probability"
            ].cpu()
        )

        all_four_probabilities.append(
            probabilities[
                "four_probability"
            ].cpu()
        )

        all_binary_probabilities.append(
            probabilities[
                "binary_probability"
            ].cpu()
        )

        all_abnormal_probabilities.append(
            probabilities[
                "abnormal_probability"
            ].cpu()
        )

        all_hierarchical_probabilities.append(
            probabilities[
                "hierarchical_probability"
            ].cpu()
        )

        all_binary_confidences.append(
            probabilities[
                "binary_confidence"
            ].cpu()
        )

        all_hierarchical_weights.append(
            probabilities[
                "hierarchical_weight"
            ].cpu()
        )

    return {
        "labels": torch.cat(
            all_labels
        ).numpy(),

        "final_probability": torch.cat(
            all_final_probabilities
        ).numpy(),

        "four_probability": torch.cat(
            all_four_probabilities
        ).numpy(),

        "binary_probability": torch.cat(
            all_binary_probabilities
        ).numpy(),

        "abnormal_probability": torch.cat(
            all_abnormal_probabilities
        ).numpy(),

        "hierarchical_probability": torch.cat(
            all_hierarchical_probabilities
        ).numpy(),

        "binary_confidence": torch.cat(
            all_binary_confidences
        ).numpy(),

        "hierarchical_weight": torch.cat(
            all_hierarchical_weights
        ).numpy(),
    }


# ============================================================
# 18. ICBHI指标
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

        "binary_accuracy": float(
            accuracy_score(
                binary_true,
                binary_pred,
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
# 19. 评估全部预测分支
# ============================================================
def evaluate_from_probabilities(
    collected,
):
    y_true = collected[
        "labels"
    ]

    final_prediction = np.argmax(
        collected[
            "final_probability"
        ],
        axis=1,
    )

    four_prediction = np.argmax(
        collected[
            "four_probability"
        ],
        axis=1,
    )

    hierarchical_prediction = np.argmax(
        collected[
            "hierarchical_probability"
        ],
        axis=1,
    )

    binary_prediction = np.argmax(
        collected[
            "binary_probability"
        ],
        axis=1,
    )

    abnormal_mask = (
        y_true > 0
    )

    abnormal_true = (
        y_true[
            abnormal_mask
        ]
        - 1
    )

    abnormal_prediction = np.argmax(
        collected[
            "abnormal_probability"
        ][abnormal_mask],
        axis=1,
    )

    binary_true = (
        y_true > 0
    ).astype(
        np.int64
    )

    auxiliary = {
        "binary_head_accuracy": float(
            accuracy_score(
                binary_true,
                binary_prediction,
            )
            * 100.0
        ),

        "binary_head_cm": confusion_matrix(
            binary_true,
            binary_prediction,
            labels=[
                0,
                1,
            ],
        ),

        "abnormal_head_accuracy": float(
            accuracy_score(
                abnormal_true,
                abnormal_prediction,
            )
            * 100.0
        ),

        "abnormal_head_recalls": recall_score(
            abnormal_true,
            abnormal_prediction,
            labels=[
                0,
                1,
                2,
            ],
            average=None,
            zero_division=0,
        ),

        "abnormal_head_cm": confusion_matrix(
            abnormal_true,
            abnormal_prediction,
            labels=[
                0,
                1,
                2,
            ],
        ),

        "mean_binary_confidence": float(
            collected[
                "binary_confidence"
            ].mean()
        ),

        "mean_hierarchical_weight": float(
            collected[
                "hierarchical_weight"
            ].mean()
        ),

        "min_hierarchical_weight": float(
            collected[
                "hierarchical_weight"
            ].min()
        ),

        "max_hierarchical_weight": float(
            collected[
                "hierarchical_weight"
            ].max()
        ),
    }

    return {
        "final": calculate_metrics(
            y_true,
            final_prediction,
        ),

        "four_only": calculate_metrics(
            y_true,
            four_prediction,
        ),

        "hierarchical_only": calculate_metrics(
            y_true,
            hierarchical_prediction,
        ),

        "auxiliary": auxiliary,
    }


# ============================================================
# 20. 最佳模型选择值
# ============================================================
def calculate_selection_value(
    metrics,
    cfg,
) -> float:
    return (
        float(
            cfg[
                "SELECTION_SCORE_WEIGHT"
            ]
        )
        * metrics["score"]

        + float(
            cfg[
                "SELECTION_MACRO_F1_WEIGHT"
            ]
        )
        * metrics["macro_f1"]
        * 100.0
    )


# ============================================================
# 21. 形状测试
# ============================================================
@torch.no_grad()
def shape_test(
    loader,
    model,
    device,
    cfg,
) -> None:
    model.eval()

    x, _ = next(
        iter(loader)
    )

    x = x[:2].to(
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

    outputs = model(x)

    probabilities = (
        model.build_probabilities(
            outputs=outputs,

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
            outputs["four_logits"].shape
        ),
        tuple(
            outputs["binary_logits"].shape
        ),
        tuple(
            outputs["abnormal_logits"].shape
        ),
    )

    print(
        "[Shape] Final Probability:",
        tuple(
            probabilities[
                "final_probability"
            ].shape
        ),
    )

    print(
        "[Shape] Hierarchical Weight:",
        probabilities[
            "hierarchical_weight"
        ].detach().cpu().flatten().tolist(),
    )

    assert tuple(
        x.shape[1:]
    ) == (
        1,
        798,
        128,
    )

    assert tuple(
        stem_map.shape[1:]
    ) == (
        64,
        399,
        64,
    )

    assert tuple(
        stage1_map.shape[1:]
    ) == (
        96,
        200,
        32,
    )

    assert tuple(
        stage2_map.shape[1:]
    ) == (
        160,
        100,
        16,
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

    assert tuple(
        probabilities[
            "final_probability"
        ].shape[1:]
    ) == (
        4,
    )

    print(
        "[PASS] D6软动态层级模型连接成功。",
        flush=True,
    )


# ============================================================
# 22. 打印指标
# ============================================================
def print_metric_block(
    title: str,
    metrics,
) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)

    print(
        f"ICBHI Score: "
        f"{metrics['score']:.4f}"
    )

    print(
        f"Specificity: "
        f"{metrics['sp']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{metrics['se']:.4f}"
    )

    print(
        f"Four-class Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Binary Accuracy: "
        f"{metrics['binary_accuracy']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        "Recall "
        "[Normal, Crackle, Wheeze, Both]:",
        np.round(
            metrics["recalls"],
            4,
        ).tolist(),
    )

    print(
        "PredCount:",
        metrics[
            "pred_counts"
        ].tolist(),
    )

    print()
    print(
        "Four-class confusion matrix:"
    )
    print(
        metrics["four_cm"]
    )

    print()
    print(
        "Binary confusion matrix:"
    )
    print(
        metrics["binary_cm"]
    )


def print_complete_evaluation(
    title: str,
    evaluation,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print_metric_block(
        "FINAL SOFT DYNAMIC FUSION",
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

    auxiliary = evaluation[
        "auxiliary"
    ]

    print()
    print("-" * 80)
    print("AUXILIARY HEAD RESULTS")
    print("-" * 80)

    print(
        f"Binary Head Accuracy: "
        f"{auxiliary['binary_head_accuracy']:.4f}"
    )

    print(
        "Binary Head Confusion Matrix:"
    )

    print(
        auxiliary[
            "binary_head_cm"
        ]
    )

    print()

    print(
        f"Abnormal Head Accuracy: "
        f"{auxiliary['abnormal_head_accuracy']:.4f}"
    )

    print(
        "Abnormal Head Recall "
        "[Crackle, Wheeze, Both]:",
        np.round(
            auxiliary[
                "abnormal_head_recalls"
            ],
            4,
        ).tolist(),
    )

    print(
        "Abnormal Head Confusion Matrix:"
    )

    print(
        auxiliary[
            "abnormal_head_cm"
        ]
    )

    print()

    print(
        f"Mean Binary Confidence: "
        f"{auxiliary['mean_binary_confidence']:.4f}"
    )

    print(
        f"Mean Hierarchical Weight: "
        f"{auxiliary['mean_hierarchical_weight']:.4f}"
    )

    print(
        f"Hierarchical Weight Range: "
        f"{auxiliary['min_hierarchical_weight']:.4f}"
        f" - "
        f"{auxiliary['max_hierarchical_weight']:.4f}"
    )


# ============================================================
# 23. 通用训练流程
# ============================================================
def run_training(
    model,
    train_loader,
    optimizer,
    base_learning_rates,
    minimum_learning_rates,
    loss_weights,
    device,
    use_amp: bool,
    cfg,
    total_epochs: int,
    validation_loader=None,
    best_checkpoint_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
):
    scaler = make_scaler(
        use_amp
    )

    history = []
    best_result = None

    for epoch in range(
        1,
        total_epochs + 1,
    ):
        epoch_start_time = time.time()

        current_learning_rates = (
            set_epoch_lrs(
                optimizer=optimizer,

                base_lrs=(
                    base_learning_rates
                ),

                minimum_lrs=(
                    minimum_learning_rates
                ),

                epoch=epoch,

                # 完整训练阶段仍沿用50轮学习率轨迹，
                # 使前N轮与开发阶段基本一致
                schedule_total_epochs=int(
                    cfg["EPOCHS"]
                ),

                warmup_epochs=int(
                    cfg["WARMUP_EPOCHS"]
                ),
            )
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

        history_row = {
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

            "consistency_loss": (
                train_result[
                    "consistency_loss"
                ]
            ),

            "frontend_lr": (
                current_learning_rates[0]
            ),

            "encoder_lr": (
                current_learning_rates[1]
            ),

            "head_lr": (
                current_learning_rates[2]
            ),

            "dtf_alpha": (
                model.get_dtf_alpha()
            ),

            "seconds": elapsed_time,
        }

        # ----------------------------------------------------
        # 内部验证阶段
        # --------------------------------------------------------
        if validation_loader is not None:
            collected = collect_probabilities(
                loader=validation_loader,

                model=model,

                device=device,

                use_amp=use_amp,

                cfg=cfg,
            )

            evaluation = (
                evaluate_from_probabilities(
                    collected
                )
            )

            validation_metrics = evaluation[
                "final"
            ]

            selection_value = (
                calculate_selection_value(
                    validation_metrics,
                    cfg,
                )
            )

            history_row.update(
                {
                    "val_selection": (
                        selection_value
                    ),

                    "val_score": (
                        validation_metrics[
                            "score"
                        ]
                    ),

                    "val_sp": (
                        validation_metrics[
                            "sp"
                        ]
                    ),

                    "val_se": (
                        validation_metrics[
                            "se"
                        ]
                    ),

                    "val_accuracy": (
                        validation_metrics[
                            "accuracy"
                        ]
                    ),

                    "val_binary_accuracy": (
                        validation_metrics[
                            "binary_accuracy"
                        ]
                    ),

                    "val_macro_f1": (
                        validation_metrics[
                            "macro_f1"
                        ]
                    ),

                    "val_four_only_score": (
                        evaluation[
                            "four_only"
                        ]["score"]
                    ),

                    "val_hierarchical_only_score": (
                        evaluation[
                            "hierarchical_only"
                        ]["score"]
                    ),
                }
            )

            can_select = (
                epoch
                >= int(
                    cfg[
                        "BEST_EPOCH_START"
                    ]
                )
            )

            is_best = False

            if can_select:
                if best_result is None:
                    is_best = True

                else:
                    current_key = (
                        selection_value,

                        validation_metrics[
                            "score"
                        ],

                        validation_metrics[
                            "macro_f1"
                        ],
                    )

                    best_key = (
                        best_result[
                            "selection_value"
                        ],

                        best_result[
                            "metrics"
                        ]["score"],

                        best_result[
                            "metrics"
                        ]["macro_f1"],
                    )

                    if current_key > best_key:
                        is_best = True

            if is_best:
                best_result = {
                    "epoch": epoch,

                    "selection_value": float(
                        selection_value
                    ),

                    "metrics": deepcopy(
                        validation_metrics
                    ),

                    "evaluation": deepcopy(
                        evaluation
                    ),

                    "model_state": (
                        state_dict_to_cpu(
                            model
                        )
                    ),
                }

                if best_checkpoint_path is not None:
                    torch.save(
                        {
                            "epoch": epoch,

                            "selection_value": float(
                                selection_value
                            ),

                            "model_state": (
                                best_result[
                                    "model_state"
                                ]
                            ),

                            "metrics": (
                                best_result[
                                    "metrics"
                                ]
                            ),

                            "config": deepcopy(
                                cfg
                            ),
                        },
                        best_checkpoint_path,
                    )

            best_epoch_text = (
                "-"
                if best_result is None
                else str(
                    best_result[
                        "epoch"
                    ]
                )
            )

            best_selection_text = (
                "-"
                if best_result is None
                else (
                    f"{best_result['selection_value']:.2f}"
                )
            )

            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{total_epochs} | "

                f"Train "
                f"{train_result['total_loss']:.4f} | "

                f"ValScore "
                f"{validation_metrics['score']:.2f} | "

                f"SP "
                f"{validation_metrics['sp']:.2f} | "

                f"SE "
                f"{validation_metrics['se']:.2f} | "

                f"Acc "
                f"{validation_metrics['accuracy']:.2f} | "

                f"MacroF1 "
                f"{validation_metrics['macro_f1']:.4f} | "

                f"Select "
                f"{selection_value:.2f} | "

                f"BestEpoch "
                f"{best_epoch_text} | "

                f"BestSelect "
                f"{best_selection_text} | "

                f"{elapsed_time:.1f}s",
                flush=True,
            )

        # ----------------------------------------------------
        # 完整官方训练阶段
        # --------------------------------------------------------
        else:
            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{total_epochs} | "

                f"Total "
                f"{train_result['total_loss']:.4f} | "

                f"Four "
                f"{train_result['four_loss']:.4f} | "

                f"Bin "
                f"{train_result['binary_loss']:.4f} | "

                f"Abn "
                f"{train_result['abnormal_loss']:.4f} | "

                f"Cons "
                f"{train_result['consistency_loss']:.4f} | "

                f"Alpha "
                f"{model.get_dtf_alpha():.4f} | "

                f"LR "
                f"{current_learning_rates[0]:.8f}/"
                f"{current_learning_rates[1]:.8f}/"
                f"{current_learning_rates[2]:.8f} | "

                f"{elapsed_time:.1f}s",
                flush=True,
            )

        history.append(
            history_row
        )

        if history_path is not None:
            pd.DataFrame(
                history
            ).to_csv(
                history_path,
                index=False,
            )

    if (
        validation_loader is not None
        and best_result is None
    ):
        raise RuntimeError(
            "未选择到最佳验证模型，"
            "请检查BEST_EPOCH_START设置。"
        )

    return best_result, history


# ============================================================
# 24. 主函数
# ============================================================
def main() -> None:
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
        bool(cfg["REQUIRE_MAMBA"])
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm导入失败，"
            "不能进行正式训练。"
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

    if not train_csv.exists():
        raise FileNotFoundError(
            train_csv
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            test_csv
        )

    official_train_dataframe = (
        pd.read_csv(
            train_csv
        )
    )

    official_test_dataframe = (
        pd.read_csv(
            test_csv
        )
    )

    official_train_dataframe[
        "label"
    ] = official_train_dataframe[
        "label"
    ].astype(int)

    official_test_dataframe[
        "label"
    ] = official_test_dataframe[
        "label"
    ].astype(int)

    # --------------------------------------------------------
    # 患者级内部划分
    # --------------------------------------------------------
    (
        internal_train_dataframe,
        internal_validation_dataframe,
    ) = patient_level_split(
        dataframe=(
            official_train_dataframe
        ),

        val_ratio=float(
            cfg["VAL_RATIO"]
        ),

        seed=int(
            cfg["SEED"]
        ),
    )

    print()
    print(
        "[Protocol] 官方测试集不参与模型选择。"
    )

    print(
        "[Protocol] 内部验证只选择最佳Epoch，"
        "不搜索、不迁移二分类阈值。"
    )

    print(
        "[Protocol] 最终推理使用固定的"
        "0.10~0.35软动态融合。"
    )

    print(
        "[Internal Train] samples:",
        len(
            internal_train_dataframe
        ),
        "| counts:",
        np.bincount(
            internal_train_dataframe[
                "label"
            ],
            minlength=4,
        ).tolist(),
        "| patients:",
        internal_train_dataframe[
            "_patient_group"
        ].nunique(),
    )

    print(
        "[Internal Validation] samples:",
        len(
            internal_validation_dataframe
        ),
        "| counts:",
        np.bincount(
            internal_validation_dataframe[
                "label"
            ],
            minlength=4,
        ).tolist(),
        "| patients:",
        internal_validation_dataframe[
            "_patient_group"
        ].nunique(),
    )

    print(
        "[Official Test] samples:",
        len(
            official_test_dataframe
        ),
        "| counts:",
        np.bincount(
            official_test_dataframe[
                "label"
            ],
            minlength=4,
        ).tolist(),
    )

    save_directory = Path(
        str(cfg["SAVE_DIR"])
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_best_path = (
        save_directory
        / "development_best.pth"
    )

    development_history_path = (
        save_directory
        / "development_history.csv"
    )

    full_train_history_path = (
        save_directory
        / "full_train_history.csv"
    )

    final_model_path = (
        save_directory
        / "final_model.pth"
    )

    # ========================================================
    # 第一阶段：患者级内部验证
    # ========================================================
    internal_train_dataset = FbankDataset(
        dataframe=(
            internal_train_dataframe
        ),

        base_directory=root,

        cfg=cfg,

        training=True,
    )

    internal_validation_dataset = FbankDataset(
        dataframe=(
            internal_validation_dataframe
        ),

        base_directory=root,

        cfg=cfg,

        training=False,
    )

    internal_train_loader = make_loader(
        dataset=(
            internal_train_dataset
        ),

        cfg=cfg,

        device=device,

        shuffle=True,
    )

    internal_validation_loader = make_loader(
        dataset=(
            internal_validation_dataset
        ),

        cfg=cfg,

        device=device,

        shuffle=False,
    )

    development_model = build_model(
        cfg,
        device,
    )

    shape_test(
        loader=internal_train_loader,

        model=development_model,

        device=device,

        cfg=cfg,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in development_model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in development_model.parameters()
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

    (
        development_optimizer,
        development_base_lrs,
        development_minimum_lrs,
    ) = build_optimizer(
        development_model,
        cfg,
    )

    loss_weights = build_loss_weights(
        cfg,
        device,
    )

    print()
    print("=" * 100)
    print(
        "D6 DEVELOPMENT: "
        "PATIENT-LEVEL INTERNAL VALIDATION"
    )
    print("=" * 100)

    best_result, _ = run_training(
        model=development_model,

        train_loader=(
            internal_train_loader
        ),

        optimizer=(
            development_optimizer
        ),

        base_learning_rates=(
            development_base_lrs
        ),

        minimum_learning_rates=(
            development_minimum_lrs
        ),

        loss_weights=loss_weights,

        device=device,

        use_amp=use_amp,

        cfg=cfg,

        total_epochs=int(
            cfg["EPOCHS"]
        ),

        validation_loader=(
            internal_validation_loader
        ),

        best_checkpoint_path=(
            development_best_path
        ),

        history_path=(
            development_history_path
        ),
    )

    print_complete_evaluation(
        title=(
            "BEST INTERNAL VALIDATION RESULT"
        ),

        evaluation=best_result[
            "evaluation"
        ],
    )

    print()
    print(
        f"Selected best epoch: "
        f"{best_result['epoch']}"
    )

    print(
        f"Selected validation value: "
        f"{best_result['selection_value']:.4f}"
    )

    # 释放开发阶段显存
    del development_model
    del development_optimizer
    del internal_train_loader
    del internal_validation_loader

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ========================================================
    # 第二阶段：完整官方训练集重新训练
    # ========================================================
    print()
    print("=" * 100)
    print(
        "D6 FINAL RETRAIN: "
        "FULL OFFICIAL TRAIN SET"
    )
    print("=" * 100)

    set_seed(
        int(cfg["SEED"])
    )

    full_train_dataset = FbankDataset(
        dataframe=(
            official_train_dataframe
        ),

        base_directory=root,

        cfg=cfg,

        training=True,
    )

    official_test_dataset = FbankDataset(
        dataframe=(
            official_test_dataframe
        ),

        base_directory=root,

        cfg=cfg,

        training=False,
    )

    full_train_loader = make_loader(
        dataset=full_train_dataset,

        cfg=cfg,

        device=device,

        shuffle=True,
    )

    official_test_loader = make_loader(
        dataset=official_test_dataset,

        cfg=cfg,

        device=device,

        shuffle=False,
    )

    final_model = build_model(
        cfg,
        device,
    )

    shape_test(
        loader=full_train_loader,

        model=final_model,

        device=device,

        cfg=cfg,
    )

    (
        final_optimizer,
        final_base_lrs,
        final_minimum_lrs,
    ) = build_optimizer(
        final_model,
        cfg,
    )

    final_loss_weights = build_loss_weights(
        cfg,
        device,
    )

    selected_epoch = int(
        best_result["epoch"]
    )

    run_training(
        model=final_model,

        train_loader=full_train_loader,

        optimizer=final_optimizer,

        base_learning_rates=(
            final_base_lrs
        ),

        minimum_learning_rates=(
            final_minimum_lrs
        ),

        loss_weights=(
            final_loss_weights
        ),

        device=device,

        use_amp=use_amp,

        cfg=cfg,

        total_epochs=selected_epoch,

        validation_loader=None,

        best_checkpoint_path=None,

        history_path=(
            full_train_history_path
        ),
    )

    # ========================================================
    # 第三阶段：官方测试集最终测试一次
    # ========================================================
    test_collected = collect_probabilities(
        loader=official_test_loader,

        model=final_model,

        device=device,

        use_amp=use_amp,

        cfg=cfg,
    )

    final_evaluation = (
        evaluate_from_probabilities(
            test_collected
        )
    )

    print_complete_evaluation(
        title=(
            "FINAL OFFICIAL TEST RESULT"
        ),

        evaluation=final_evaluation,
    )

    print()
    print(
        f"Selected epoch: "
        f"{selected_epoch}"
    )

    print(
        f"Dynamic hierarchical weight: "
        f"{cfg['MIN_HIERARCHICAL_WEIGHT']}"
        f" ~ "
        f"{cfg['MAX_HIERARCHICAL_WEIGHT']}"
    )

    # ========================================================
    # 保存最终模型
    # ========================================================
    final_result = final_evaluation[
        "final"
    ]

    four_only_result = final_evaluation[
        "four_only"
    ]

    hierarchical_only_result = (
        final_evaluation[
            "hierarchical_only"
        ]
    )

    auxiliary_result = (
        final_evaluation[
            "auxiliary"
        ]
    )

    torch.save(
        {
            "epoch": selected_epoch,

            "model_state": (
                state_dict_to_cpu(
                    final_model
                )
            ),

            "config": deepcopy(
                cfg
            ),

            "development_selection_value": (
                best_result[
                    "selection_value"
                ]
            ),

            "development_metrics": {
                "score": (
                    best_result[
                        "metrics"
                    ]["score"]
                ),

                "sp": (
                    best_result[
                        "metrics"
                    ]["sp"]
                ),

                "se": (
                    best_result[
                        "metrics"
                    ]["se"]
                ),

                "accuracy": (
                    best_result[
                        "metrics"
                    ]["accuracy"]
                ),

                "binary_accuracy": (
                    best_result[
                        "metrics"
                    ]["binary_accuracy"]
                ),

                "macro_f1": (
                    best_result[
                        "metrics"
                    ]["macro_f1"]
                ),
            },

            "final_test_metrics": {
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

                "binary_accuracy": (
                    final_result[
                        "binary_accuracy"
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
            },

            "four_only_metrics": {
                "score": (
                    four_only_result[
                        "score"
                    ]
                ),

                "sp": (
                    four_only_result[
                        "sp"
                    ]
                ),

                "se": (
                    four_only_result[
                        "se"
                    ]
                ),

                "accuracy": (
                    four_only_result[
                        "accuracy"
                    ]
                ),

                "macro_f1": (
                    four_only_result[
                        "macro_f1"
                    ]
                ),

                "four_cm": (
                    four_only_result[
                        "four_cm"
                    ].tolist()
                ),
            },

            "hierarchical_only_metrics": {
                "score": (
                    hierarchical_only_result[
                        "score"
                    ]
                ),

                "sp": (
                    hierarchical_only_result[
                        "sp"
                    ]
                ),

                "se": (
                    hierarchical_only_result[
                        "se"
                    ]
                ),

                "accuracy": (
                    hierarchical_only_result[
                        "accuracy"
                    ]
                ),

                "macro_f1": (
                    hierarchical_only_result[
                        "macro_f1"
                    ]
                ),

                "four_cm": (
                    hierarchical_only_result[
                        "four_cm"
                    ].tolist()
                ),
            },

            "auxiliary_metrics": {
                "binary_head_accuracy": (
                    auxiliary_result[
                        "binary_head_accuracy"
                    ]
                ),

                "binary_head_cm": (
                    auxiliary_result[
                        "binary_head_cm"
                    ].tolist()
                ),

                "abnormal_head_accuracy": (
                    auxiliary_result[
                        "abnormal_head_accuracy"
                    ]
                ),

                "abnormal_head_recalls": (
                    auxiliary_result[
                        "abnormal_head_recalls"
                    ].tolist()
                ),

                "abnormal_head_cm": (
                    auxiliary_result[
                        "abnormal_head_cm"
                    ].tolist()
                ),

                "mean_binary_confidence": (
                    auxiliary_result[
                        "mean_binary_confidence"
                    ]
                ),

                "mean_hierarchical_weight": (
                    auxiliary_result[
                        "mean_hierarchical_weight"
                    ]
                ),

                "min_hierarchical_weight": (
                    auxiliary_result[
                        "min_hierarchical_weight"
                    ]
                ),

                "max_hierarchical_weight": (
                    auxiliary_result[
                        "max_hierarchical_weight"
                    ]
                ),
            },
        },
        final_model_path,
    )

    print()
    print(
        "Development best:",
        development_best_path,
    )

    print(
        "Development history:",
        development_history_path,
    )

    print(
        "Full train history:",
        full_train_history_path,
    )

    print(
        "Final model:",
        final_model_path,
    )


if __name__ == "__main__":
    main()