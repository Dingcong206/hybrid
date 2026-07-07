#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
        "checkpoints_d4_1_balanced_fusion_seed42"
    ),

    # --------------------------------------------------------
    # 训练设置
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

    # 第5轮之后才开始选择最佳模型
    "BEST_EPOCH_START": 5,

    # --------------------------------------------------------
    # 输入尺寸
    # --------------------------------------------------------
    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

    # --------------------------------------------------------
    # 模型参数：必须与D4 model.py一致
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

    "BINARY_RESIDUAL_SCALE": 0.50,

    # --------------------------------------------------------
    # 多任务损失
    #
    # 原D4：
    # Four=1.00
    # Binary=0.25
    # Abnormal=0.75
    #
    # D4.1：
    # 降低Binary干扰，加强异常类别监督
    # --------------------------------------------------------
    "FOUR_LOSS_WEIGHT": 1.00,
    "BINARY_LOSS_WEIGHT": 0.15,
    "ABNORMAL_LOSS_WEIGHT": 1.00,

    # --------------------------------------------------------
    # 类别权重
    # --------------------------------------------------------
    "USE_FOUR_CLASS_WEIGHTS": True,
    "USE_BINARY_CLASS_WEIGHTS": False,
    "USE_ABNORMAL_CLASS_WEIGHTS": True,

    # Normal / Crackle / Wheeze / Both
    "FOUR_MANUAL_WEIGHTS": [
        1.00,
        1.00,
        1.10,
        1.20,
    ],

    # Crackle / Wheeze / Both
    "ABNORMAL_MANUAL_WEIGHTS": [
        1.00,
        1.05,
        1.20,
    ],

    # --------------------------------------------------------
    # Label smoothing
    # --------------------------------------------------------
    "FOUR_LABEL_SMOOTHING": 0.00,
    "BINARY_LABEL_SMOOTHING": 0.00,
    "ABNORMAL_LABEL_SMOOTHING": 0.00,

    # --------------------------------------------------------
    # 动态层级融合
    #
    # 原D4：0.15～0.60
    # 当前降低为0.05～0.30
    # --------------------------------------------------------
    "MIN_HIERARCHICAL_WEIGHT": 0.05,
    "MAX_HIERARCHICAL_WEIGHT": 0.30,

    # --------------------------------------------------------
    # 验证集搜索固定融合比例
    #
    # 数值表示Four Head在最终融合中的权重
    #
    # 例如：
    # four_weight=0.80
    # hierarchical_weight=0.20
    #
    # None表示使用0.05～0.30动态融合
    # --------------------------------------------------------
    "FUSION_CANDIDATES": [
        None,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ],

    # --------------------------------------------------------
    # SpecAugment
    #
    # 原D4：80 / 16
    # 当前：64 / 12
    # --------------------------------------------------------
    "USE_SPECAUGMENT": True,

    "TIME_MASK_MAX": 64,
    "FREQ_MASK_MAX": 12,

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
    # Score为主要目标；
    # Macro-F1只用于Score相同时的辅助排序。
    # --------------------------------------------------------
    "PRINT_INTERVAL": 50,
}


FusionCandidate = Optional[float]


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
# 5. State Dict复制到CPU
# ============================================================
def state_dict_to_cpu(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }


# ============================================================
# 6. 患者ID推断
# ============================================================
def infer_patient_id(path_value: str) -> str:
    """
    例如：
        101_1b1_Al_sc_Meditron_0.npy

    第一个下划线之前的101作为患者ID。
    """

    filename = Path(str(path_value)).stem
    return filename.split("_")[0]


def add_patient_group(
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
                dataframe[column].astype(str)
            )
            return dataframe

    dataframe["_patient_group"] = (
        dataframe["fbank_path"]
        .astype(str)
        .map(infer_patient_id)
    )

    return dataframe


# ============================================================
# 7. 患者级内部划分
# ============================================================
def patient_level_split(
    dataframe: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = add_patient_group(dataframe)

    labels = dataframe["label"].to_numpy(
        dtype=np.int64
    )

    groups = dataframe[
        "_patient_group"
    ].to_numpy()

    if len(np.unique(groups)) < 2:
        raise RuntimeError(
            "检测到的患者数量不足，无法执行患者级划分。"
        )

    full_counts = np.bincount(
        labels,
        minlength=4,
    ).astype(np.float64)

    full_distribution = (
        full_counts
        / full_counts.sum()
    )

    best_train_dataframe = None
    best_validation_dataframe = None
    best_difference = float("inf")

    # 搜索类别比例相对均衡的患者级划分
    for offset in range(500):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_ratio,
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

        # 两个集合都必须包含四个类别
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

        # 禁止患者泄漏
        if train_patients & validation_patients:
            continue

        validation_distribution = (
            validation_counts.astype(np.float64)
            / validation_counts.sum()
        )

        distribution_difference = np.abs(
            validation_distribution
            - full_distribution
        ).sum()

        size_difference = abs(
            len(validation_dataframe)
            / len(dataframe)
            - validation_ratio
        )

        total_difference = float(
            distribution_difference
            + size_difference
        )

        if total_difference < best_difference:
            best_difference = total_difference
            best_train_dataframe = train_dataframe
            best_validation_dataframe = validation_dataframe

    if (
        best_train_dataframe is None
        or best_validation_dataframe is None
    ):
        raise RuntimeError(
            "无法生成同时包含四类的患者级内部验证集。"
        )

    return (
        best_train_dataframe,
        best_validation_dataframe,
    )


# ============================================================
# 8. Warmup + Cosine学习率
# ============================================================
def set_epoch_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: List[float],
    minimum_lrs: List[float],
    epoch: int,
    schedule_total_epochs: int,
    warmup_epochs: int,
) -> List[float]:
    if epoch <= warmup_epochs:
        warmup_ratio = (
            0.20
            + 0.80
            * epoch
            / max(warmup_epochs, 1)
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
    num_time_masks: int,
    num_frequency_masks: int,
) -> torch.Tensor:
    if fbank.ndim != 2:
        raise ValueError(
            "SpecAugment输入必须为[T,F]，"
            f"当前={tuple(fbank.shape)}。"
        )

    x = fbank.clone()

    time_frames = int(x.shape[0])
    frequency_bins = int(x.shape[1])

    mask_value = x.mean()

    # --------------------------------------------------------
    # Time Mask
    # --------------------------------------------------------
    for _ in range(max(num_time_masks, 0)):
        if time_mask_max <= 0:
            break

        mask_width = random.randint(
            0,
            min(time_mask_max, time_frames),
        )

        if mask_width > 0:
            mask_start = random.randint(
                0,
                time_frames - mask_width,
            )

            x[
                mask_start:
                mask_start + mask_width,
                :
            ] = mask_value

    # --------------------------------------------------------
    # Frequency Mask
    # --------------------------------------------------------
    for _ in range(max(num_frequency_masks, 0)):
        if frequency_mask_max <= 0:
            break

        mask_width = random.randint(
            0,
            min(
                frequency_mask_max,
                frequency_bins,
            ),
        )

        if mask_width > 0:
            mask_start = random.randint(
                0,
                frequency_bins - mask_width,
            )

            x[
                :,
                mask_start:
                mask_start + mask_width,
            ] = mask_value

    return x


# ============================================================
# 10. Dataset
# ============================================================
class FbankDataset(Dataset):
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
                f"数据表缺少必要列："
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
        fbank_path = Path(str(raw_path))

        if fbank_path.exists():
            return fbank_path

        candidate_path = (
            self.base_directory
            / fbank_path
        )

        if candidate_path.exists():
            return candidate_path

        raise FileNotFoundError(
            f"找不到Fbank文件：{raw_path}"
        )

    def __getitem__(
        self,
        index: int,
    ):
        row = self.dataframe.iloc[index]

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
                f"Fbank中存在NaN或Inf：{fbank_path}"
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

        # [798,128] -> [1,798,128]
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
    dataset: Dataset,
    cfg,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    workers = int(cfg["NUM_WORKERS"])

    loader_arguments = {
        "dataset": dataset,
        "batch_size": int(cfg["BATCH_SIZE"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "drop_last": False,
    }

    if workers > 0:
        loader_arguments["prefetch_factor"] = 2

    return DataLoader(**loader_arguments)


# ============================================================
# 12. 构建模型
# ============================================================
def build_model(
    cfg,
    device: torch.device,
) -> DTFHybridModel:
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
            cfg["BINARY_RESIDUAL_SCALE"]
        ),
    ).to(device)

    return model


# ============================================================
# 13. 构建优化器
# ============================================================
def build_optimizer(
    model: DTFHybridModel,
    cfg,
):
    head_parameters = (
        list(model.four_head.parameters())
        + list(model.binary_head.parameters())
        + list(model.abnormal_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.frontend.parameters(),
                "lr": float(cfg["FRONTEND_LR"]),
            },
            {
                "params": model.encoder.parameters(),
                "lr": float(cfg["ENCODER_LR"]),
            },
            {
                "params": head_parameters,
                "lr": float(cfg["HEAD_LR"]),
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

    return optimizer, base_lrs, minimum_lrs


# ============================================================
# 14. 构建损失类别权重
# ============================================================
def build_loss_weights(
    cfg,
    device: torch.device,
):
    if bool(
        cfg["USE_FOUR_CLASS_WEIGHTS"]
    ):
        four_weight = torch.tensor(
            cfg["FOUR_MANUAL_WEIGHTS"],
            dtype=torch.float32,
            device=device,
        )
    else:
        four_weight = None

    if bool(
        cfg["USE_BINARY_CLASS_WEIGHTS"]
    ):
        raise NotImplementedError(
            "当前版本不建议给二分类设置类别权重。"
        )
    else:
        binary_weight = None

    if bool(
        cfg["USE_ABNORMAL_CLASS_WEIGHTS"]
    ):
        abnormal_weight = torch.tensor(
            cfg["ABNORMAL_MANUAL_WEIGHTS"],
            dtype=torch.float32,
            device=device,
        )
    else:
        abnormal_weight = None

    if (
        four_weight is not None
        and four_weight.numel() != 4
    ):
        raise ValueError(
            "FOUR_MANUAL_WEIGHTS必须包含4个值。"
        )

    if (
        abnormal_weight is not None
        and abnormal_weight.numel() != 3
    ):
        raise ValueError(
            "ABNORMAL_MANUAL_WEIGHTS必须包含3个值。"
        )

    print(
        "[Loss] Four-class weight:",
        (
            None
            if four_weight is None
            else four_weight.detach().cpu().tolist()
        ),
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
            else abnormal_weight.detach().cpu().tolist()
        ),
    )

    return {
        "four": four_weight,
        "binary": binary_weight,
        "abnormal": abnormal_weight,
    }


# ============================================================
# 15. 多任务损失
# ============================================================
def calculate_multitask_loss(
    outputs,
    labels,
    loss_weights,
    cfg,
):
    # --------------------------------------------------------
    # 四分类损失
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
    # 二分类损失
    # Normal=0，Abnormal=1
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
    # 异常三分类损失
    # 1/2/3 -> 0/1/2
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

            weight=loss_weights["abnormal"],

            label_smoothing=float(
                cfg["ABNORMAL_LABEL_SMOOTHING"]
            ),
        )

    else:
        abnormal_loss = (
            outputs["abnormal_logits"].sum()
            * 0.0
        )

    total_loss = (
        float(cfg["FOUR_LOSS_WEIGHT"])
        * four_loss

        + float(cfg["BINARY_LOSS_WEIGHT"])
        * binary_loss

        + float(cfg["ABNORMAL_LOSS_WEIGHT"])
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
# 16. 单轮训练
# ============================================================
def train_one_epoch(
    loader: DataLoader,
    model: DTFHybridModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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

        window_start = (
            batch_index
            // accumulation_steps
        ) * accumulation_steps

        window_end = min(
            window_start + accumulation_steps,
            total_batches,
        )

        current_accumulation_size = max(
            window_end - window_start,
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
                loss_result["total_loss"]
                .detach()
                .item()
            )
            * batch_size
        )

        accumulated_four_loss += (
            float(
                loss_result["four_loss"]
                .detach()
                .item()
            )
            * batch_size
        )

        accumulated_binary_loss += (
            float(
                loss_result["binary_loss"]
                .detach()
                .item()
            )
            * batch_size
        )

        if abnormal_count > 0:
            accumulated_abnormal_loss += (
                float(
                    loss_result["abnormal_loss"]
                    .detach()
                    .item()
                )
                * abnormal_count
            )

        should_update = (
            completed_batches == window_end
        )

        if should_update:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                float(cfg["GRAD_CLIP"]),
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        if (
            completed_batches == 1
            or completed_batches
            % int(cfg["PRINT_INTERVAL"])
            == 0
            or completed_batches == total_batches
        ):
            elapsed = (
                time.time()
                - epoch_start_time
            )

            average_batch_time = (
                elapsed
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
            / max(total_samples, 1)
        ),

        "four_loss": (
            accumulated_four_loss
            / max(total_samples, 1)
        ),

        "binary_loss": (
            accumulated_binary_loss
            / max(total_samples, 1)
        ),

        "abnormal_loss": (
            accumulated_abnormal_loss
            / max(total_abnormal_samples, 1)
        ),
    }


# ============================================================
# 17. 指标计算
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
        labels=[0, 1, 2, 3],
    )

    normal_total = max(
        int(four_cm[0].sum()),
        1,
    )

    abnormal_total = max(
        int(four_cm[1:].sum()),
        1,
    )

    specificity = (
        100.0
        * float(four_cm[0, 0])
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

    binary_prediction = (
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

        "binary_accuracy": float(
            accuracy_score(
                binary_true,
                binary_prediction,
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
            labels=[0, 1, 2, 3],
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
            binary_prediction,
            labels=[0, 1],
        ),
    }


# ============================================================
# 18. 融合名称
# ============================================================
def fusion_name(
    candidate: FusionCandidate,
) -> str:
    if candidate is None:
        return "dynamic"

    return f"four={candidate:.2f}"


# ============================================================
# 19. 验证多个融合比例
# ============================================================
@torch.no_grad()
def evaluate_fusion_candidates(
    loader: DataLoader,
    model: DTFHybridModel,
    device: torch.device,
    cfg,
    use_amp: bool,
):
    model.eval()

    candidates: List[FusionCandidate] = list(
        cfg["FUSION_CANDIDATES"]
    )

    all_labels = []

    candidate_predictions = {
        fusion_name(candidate): []
        for candidate in candidates
    }

    all_four_predictions = []
    all_binary_predictions = []

    all_abnormal_true = []
    all_abnormal_predictions = []

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

            for candidate in candidates:
                probabilities = (
                    model.build_probabilities(
                        outputs=outputs,

                        four_weight=candidate,

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

                prediction = torch.argmax(
                    probabilities[
                        "final_probability"
                    ],
                    dim=1,
                )

                candidate_predictions[
                    fusion_name(candidate)
                ].append(
                    prediction.cpu()
                )

            four_prediction = torch.argmax(
                outputs["four_logits"],
                dim=1,
            )

            binary_prediction = torch.argmax(
                outputs["binary_logits"],
                dim=1,
            )

            abnormal_prediction = torch.argmax(
                outputs["abnormal_logits"],
                dim=1,
            )

        all_labels.append(y.cpu())

        all_four_predictions.append(
            four_prediction.cpu()
        )

        all_binary_predictions.append(
            binary_prediction.cpu()
        )

        abnormal_mask = (
            y_device > 0
        )

        if int(
            abnormal_mask.sum().item()
        ) > 0:
            all_abnormal_true.append(
                (
                    y_device[abnormal_mask]
                    - 1
                ).cpu()
            )

            all_abnormal_predictions.append(
                abnormal_prediction[
                    abnormal_mask
                ].cpu()
            )

    y_true = torch.cat(
        all_labels
    ).numpy()

    candidate_results = {}

    for candidate in candidates:
        name = fusion_name(candidate)

        prediction = torch.cat(
            candidate_predictions[name]
        ).numpy()

        candidate_results[name] = {
            "candidate": candidate,
            "metrics": calculate_metrics(
                y_true,
                prediction,
            ),
        }

    # Score优先，之后依次比较Macro-F1、SE和Accuracy
    best_name = max(
        candidate_results.keys(),
        key=lambda name: (
            candidate_results[name][
                "metrics"
            ]["score"],

            candidate_results[name][
                "metrics"
            ]["macro_f1"],

            candidate_results[name][
                "metrics"
            ]["se"],

            candidate_results[name][
                "metrics"
            ]["accuracy"],
        ),
    )

    best_result = candidate_results[
        best_name
    ]

    binary_true = (
        y_true > 0
    ).astype(np.int64)

    binary_prediction = torch.cat(
        all_binary_predictions
    ).numpy()

    abnormal_true = torch.cat(
        all_abnormal_true
    ).numpy()

    abnormal_prediction = torch.cat(
        all_abnormal_predictions
    ).numpy()

    auxiliary = {
        "four_only_metrics": calculate_metrics(
            y_true,
            torch.cat(
                all_four_predictions
            ).numpy(),
        ),

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
            labels=[0, 1],
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
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        ),

        "abnormal_head_cm": confusion_matrix(
            abnormal_true,
            abnormal_prediction,
            labels=[0, 1, 2],
        ),
    }

    return {
        "best_candidate": best_result[
            "candidate"
        ],

        "best_name": best_name,

        "best_metrics": best_result[
            "metrics"
        ],

        "all_candidates": candidate_results,

        "auxiliary": auxiliary,
    }


# ============================================================
# 20. 单个融合方式最终评估
# ============================================================
@torch.no_grad()
def evaluate_single_fusion(
    loader: DataLoader,
    model: DTFHybridModel,
    device: torch.device,
    cfg,
    use_amp: bool,
    fusion_candidate: FusionCandidate,
):
    model.eval()

    all_labels = []
    all_final_predictions = []
    all_four_predictions = []

    all_binary_predictions = []

    all_abnormal_true = []
    all_abnormal_predictions = []

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

            probabilities = model.build_probabilities(
                outputs=outputs,

                four_weight=fusion_candidate,

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

            final_prediction = torch.argmax(
                probabilities[
                    "final_probability"
                ],
                dim=1,
            )

            four_prediction = torch.argmax(
                probabilities[
                    "four_probability"
                ],
                dim=1,
            )

            binary_prediction = torch.argmax(
                probabilities[
                    "binary_probability"
                ],
                dim=1,
            )

            abnormal_prediction = torch.argmax(
                probabilities[
                    "abnormal_probability"
                ],
                dim=1,
            )

        all_labels.append(y.cpu())

        all_final_predictions.append(
            final_prediction.cpu()
        )

        all_four_predictions.append(
            four_prediction.cpu()
        )

        all_binary_predictions.append(
            binary_prediction.cpu()
        )

        abnormal_mask = (
            y_device > 0
        )

        if int(
            abnormal_mask.sum().item()
        ) > 0:
            all_abnormal_true.append(
                (
                    y_device[abnormal_mask]
                    - 1
                ).cpu()
            )

            all_abnormal_predictions.append(
                abnormal_prediction[
                    abnormal_mask
                ].cpu()
            )

    y_true = torch.cat(
        all_labels
    ).numpy()

    final_prediction = torch.cat(
        all_final_predictions
    ).numpy()

    four_prediction = torch.cat(
        all_four_predictions
    ).numpy()

    binary_prediction = torch.cat(
        all_binary_predictions
    ).numpy()

    binary_true = (
        y_true > 0
    ).astype(np.int64)

    abnormal_true = torch.cat(
        all_abnormal_true
    ).numpy()

    abnormal_prediction = torch.cat(
        all_abnormal_predictions
    ).numpy()

    return {
        "final": calculate_metrics(
            y_true,
            final_prediction,
        ),

        "four_only": calculate_metrics(
            y_true,
            four_prediction,
        ),

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
            labels=[0, 1],
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
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        ),

        "abnormal_head_cm": confusion_matrix(
            abnormal_true,
            abnormal_prediction,
            labels=[0, 1, 2],
        ),
    }


# ============================================================
# 21. 形状测试
# ============================================================
@torch.no_grad()
def shape_test(
    loader: DataLoader,
    model: DTFHybridModel,
    device: torch.device,
    cfg,
):
    model.eval()

    x, _ = next(iter(loader))

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

    probabilities = model.build_probabilities(
        outputs=outputs,

        four_weight=None,

        minimum_hierarchical_weight=float(
            cfg["MIN_HIERARCHICAL_WEIGHT"]
        ),

        maximum_hierarchical_weight=float(
            cfg["MAX_HIERARCHICAL_WEIGHT"]
        ),
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
        tuple(outputs["four_logits"].shape),
        tuple(outputs["binary_logits"].shape),
        tuple(outputs["abnormal_logits"].shape),
    )

    print(
        "[Shape] Final Probability:",
        tuple(
            probabilities[
                "final_probability"
            ].shape
        ),
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
        outputs["four_logits"].shape[1:]
    ) == (4,)

    assert tuple(
        outputs["binary_logits"].shape[1:]
    ) == (2,)

    assert tuple(
        outputs["abnormal_logits"].shape[1:]
    ) == (3,)

    print(
        "[PASS] D4.1模型连接成功。",
        flush=True,
    )


# ============================================================
# 22. 打印指标
# ============================================================
def print_metrics(
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
        f"Accuracy: "
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
        "Recall [Normal, Crackle, Wheeze, Both]:",
        np.round(
            metrics["recalls"],
            4,
        ).tolist(),
    )

    print(
        "PredCount:",
        metrics["pred_counts"].tolist(),
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
    evaluation,
    fusion_candidate: FusionCandidate,
) -> None:
    print()
    print("=" * 100)
    print("FINAL OFFICIAL TEST RESULT")
    print("=" * 100)

    print(
        "Selected fusion:",
        fusion_name(
            fusion_candidate
        ),
    )

    print_metrics(
        "FINAL FUSED PREDICTION",
        evaluation["final"],
    )

    print_metrics(
        "FOUR-CLASS HEAD ONLY",
        evaluation["four_only"],
    )

    print()
    print("-" * 80)
    print("AUXILIARY HEAD RESULTS")
    print("-" * 80)

    print(
        f"Binary Head Accuracy: "
        f"{evaluation['binary_head_accuracy']:.4f}"
    )

    print(
        "Binary Head Confusion Matrix:"
    )

    print(
        evaluation["binary_head_cm"]
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


# ============================================================
# 23. 开发阶段训练
# ============================================================
def development_training(
    model: DTFHybridModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    base_lrs: List[float],
    minimum_lrs: List[float],
    loss_weights,
    device: torch.device,
    use_amp: bool,
    cfg,
    best_checkpoint_path: Path,
    history_path: Path,
):
    scaler = make_scaler(use_amp)

    best_result = None
    history = []

    total_epochs = int(
        cfg["EPOCHS"]
    )

    for epoch in range(
        1,
        total_epochs + 1,
    ):
        current_lrs = set_epoch_lrs(
            optimizer=optimizer,
            base_lrs=base_lrs,
            minimum_lrs=minimum_lrs,
            epoch=epoch,
            schedule_total_epochs=total_epochs,
            warmup_epochs=int(
                cfg["WARMUP_EPOCHS"]
            ),
        )

        epoch_start_time = time.time()

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

        validation_result = (
            evaluate_fusion_candidates(
                loader=validation_loader,
                model=model,
                device=device,
                cfg=cfg,
                use_amp=use_amp,
            )
        )

        validation_metrics = (
            validation_result[
                "best_metrics"
            ]
        )

        elapsed_time = (
            time.time()
            - epoch_start_time
        )

        history_row = {
            "epoch": epoch,

            "train_total_loss": (
                train_result[
                    "total_loss"
                ]
            ),

            "train_four_loss": (
                train_result[
                    "four_loss"
                ]
            ),

            "train_binary_loss": (
                train_result[
                    "binary_loss"
                ]
            ),

            "train_abnormal_loss": (
                train_result[
                    "abnormal_loss"
                ]
            ),

            "val_fusion": (
                validation_result[
                    "best_name"
                ]
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

            "frontend_lr": current_lrs[0],
            "encoder_lr": current_lrs[1],
            "head_lr": current_lrs[2],

            "dtf_alpha": (
                model.get_dtf_alpha()
            ),

            "seconds": elapsed_time,
        }

        history.append(history_row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        can_select = (
            epoch
            >= int(
                cfg["BEST_EPOCH_START"]
            )
        )

        is_best = False

        if can_select:
            if best_result is None:
                is_best = True

            else:
                current_key = (
                    validation_metrics["score"],
                    validation_metrics["macro_f1"],
                    validation_metrics["se"],
                    validation_metrics["accuracy"],
                )

                best_key = (
                    best_result["metrics"]["score"],
                    best_result["metrics"]["macro_f1"],
                    best_result["metrics"]["se"],
                    best_result["metrics"]["accuracy"],
                )

                if current_key > best_key:
                    is_best = True

        if is_best:
            best_result = {
                "epoch": epoch,

                "fusion_candidate": (
                    validation_result[
                        "best_candidate"
                    ]
                ),

                "fusion_name": (
                    validation_result[
                        "best_name"
                    ]
                ),

                "metrics": deepcopy(
                    validation_metrics
                ),

                "model_state": (
                    state_dict_to_cpu(
                        model
                    )
                ),
            }

            torch.save(
                {
                    "epoch": epoch,

                    "fusion_candidate": (
                        best_result[
                            "fusion_candidate"
                        ]
                    ),

                    "fusion_name": (
                        best_result[
                            "fusion_name"
                        ]
                    ),

                    "metrics": (
                        best_result[
                            "metrics"
                        ]
                    ),

                    "model_state": (
                        best_result[
                            "model_state"
                        ]
                    ),

                    "config": deepcopy(cfg),
                },
                best_checkpoint_path,
            )

        best_epoch_text = (
            "-"
            if best_result is None
            else str(
                best_result["epoch"]
            )
        )

        best_score_text = (
            "-"
            if best_result is None
            else f"{best_result['metrics']['score']:.2f}"
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

            f"F1 "
            f"{validation_metrics['macro_f1']:.4f} | "

            f"Fusion "
            f"{validation_result['best_name']} | "

            f"BestEpoch "
            f"{best_epoch_text} | "

            f"BestScore "
            f"{best_score_text} | "

            f"Alpha "
            f"{model.get_dtf_alpha():.4f} | "

            f"{elapsed_time:.1f}s",
            flush=True,
        )

    if best_result is None:
        raise RuntimeError(
            "开发阶段未选择到最佳模型。"
        )

    return best_result


# ============================================================
# 24. 完整训练集重新训练
# ============================================================
def full_retraining(
    model: DTFHybridModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    base_lrs: List[float],
    minimum_lrs: List[float],
    loss_weights,
    device: torch.device,
    use_amp: bool,
    cfg,
    selected_epoch: int,
    history_path: Path,
):
    scaler = make_scaler(use_amp)

    history = []

    for epoch in range(
        1,
        selected_epoch + 1,
    ):
        current_lrs = set_epoch_lrs(
            optimizer=optimizer,
            base_lrs=base_lrs,
            minimum_lrs=minimum_lrs,
            epoch=epoch,

            # 保持与开发阶段一致的50轮学习率曲线
            schedule_total_epochs=int(
                cfg["EPOCHS"]
            ),

            warmup_epochs=int(
                cfg["WARMUP_EPOCHS"]
            ),
        )

        epoch_start_time = time.time()

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

                "frontend_lr": current_lrs[0],
                "encoder_lr": current_lrs[1],
                "head_lr": current_lrs[2],

                "dtf_alpha": (
                    model.get_dtf_alpha()
                ),

                "seconds": elapsed_time,
            }
        )

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{selected_epoch} | "

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

            f"{elapsed_time:.1f}s",
            flush=True,
        )


# ============================================================
# 25. 主函数
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

    official_train_dataframe = pd.read_csv(
        train_csv
    )

    official_test_dataframe = pd.read_csv(
        test_csv
    )

    official_train_dataframe["label"] = (
        official_train_dataframe["label"]
        .astype(int)
    )

    official_test_dataframe["label"] = (
        official_test_dataframe["label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # 患者级内部训练/验证划分
    # --------------------------------------------------------
    (
        internal_train_dataframe,
        internal_validation_dataframe,
    ) = patient_level_split(
        dataframe=official_train_dataframe,

        validation_ratio=float(
            cfg["VAL_RATIO"]
        ),

        seed=int(
            cfg["SEED"]
        ),
    )

    print()
    print(
        "[Protocol] 官方测试集不参与Epoch和融合权重选择。"
    )

    print(
        "[Protocol] 使用官方训练集进行患者级内部验证。"
    )

    print(
        "[Protocol] 选择最佳Epoch和融合方式后，"
        "使用完整官方训练集重新训练。"
    )

    print(
        "[Internal Train] samples:",
        len(internal_train_dataframe),

        "| counts:",
        np.bincount(
            internal_train_dataframe["label"],
            minlength=4,
        ).tolist(),

        "| patients:",
        internal_train_dataframe[
            "_patient_group"
        ].nunique(),
    )

    print(
        "[Internal Validation] samples:",
        len(internal_validation_dataframe),

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
        len(official_test_dataframe),

        "| counts:",
        np.bincount(
            official_test_dataframe["label"],
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
        dataframe=internal_train_dataframe,
        base_directory=root,
        cfg=cfg,
        training=True,
    )

    internal_validation_dataset = FbankDataset(
        dataframe=internal_validation_dataframe,
        base_directory=root,
        cfg=cfg,
        training=False,
    )

    internal_train_loader = make_loader(
        dataset=internal_train_dataset,
        cfg=cfg,
        device=device,
        shuffle=True,
    )

    internal_validation_loader = make_loader(
        dataset=internal_validation_dataset,
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
        "D4.1 DEVELOPMENT: "
        "PATIENT-LEVEL VALIDATION "
        "+ FUSION SEARCH"
    )
    print("=" * 100)

    best_result = development_training(
        model=development_model,

        train_loader=internal_train_loader,

        validation_loader=(
            internal_validation_loader
        ),

        optimizer=(
            development_optimizer
        ),

        base_lrs=(
            development_base_lrs
        ),

        minimum_lrs=(
            development_minimum_lrs
        ),

        loss_weights=loss_weights,

        device=device,

        use_amp=use_amp,

        cfg=cfg,

        best_checkpoint_path=(
            development_best_path
        ),

        history_path=(
            development_history_path
        ),
    )

    print_metrics(
        "BEST INTERNAL VALIDATION",
        best_result["metrics"],
    )

    print(
        f"Selected epoch: "
        f"{best_result['epoch']}"
    )

    print(
        f"Selected fusion: "
        f"{best_result['fusion_name']}"
    )

    selected_epoch = int(
        best_result["epoch"]
    )

    selected_fusion = (
        best_result[
            "fusion_candidate"
        ]
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
        "D4.1 FINAL RETRAIN: "
        "FULL OFFICIAL TRAIN SET"
    )
    print("=" * 100)

    set_seed(
        int(cfg["SEED"])
    )

    full_train_dataset = FbankDataset(
        dataframe=official_train_dataframe,
        base_directory=root,
        cfg=cfg,
        training=True,
    )

    official_test_dataset = FbankDataset(
        dataframe=official_test_dataframe,
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

    full_retraining(
        model=final_model,

        train_loader=full_train_loader,

        optimizer=final_optimizer,

        base_lrs=final_base_lrs,

        minimum_lrs=final_minimum_lrs,

        loss_weights=final_loss_weights,

        device=device,

        use_amp=use_amp,

        cfg=cfg,

        selected_epoch=selected_epoch,

        history_path=full_train_history_path,
    )

    # ========================================================
    # 第三阶段：官方测试集测试一次
    # ========================================================
    final_evaluation = evaluate_single_fusion(
        loader=official_test_loader,

        model=final_model,

        device=device,

        cfg=cfg,

        use_amp=use_amp,

        fusion_candidate=selected_fusion,
    )

    print_complete_evaluation(
        evaluation=final_evaluation,

        fusion_candidate=selected_fusion,
    )

    final_metrics = final_evaluation[
        "final"
    ]

    # ========================================================
    # 保存最终模型
    # ========================================================
    torch.save(
        {
            "epoch": selected_epoch,

            "fusion_candidate": selected_fusion,

            "fusion_name": fusion_name(
                selected_fusion
            ),

            "model_state": state_dict_to_cpu(
                final_model
            ),

            "config": deepcopy(cfg),

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

            "test_metrics": {
                "score": (
                    final_metrics[
                        "score"
                    ]
                ),

                "sp": (
                    final_metrics[
                        "sp"
                    ]
                ),

                "se": (
                    final_metrics[
                        "se"
                    ]
                ),

                "accuracy": (
                    final_metrics[
                        "accuracy"
                    ]
                ),

                "binary_accuracy": (
                    final_metrics[
                        "binary_accuracy"
                    ]
                ),

                "macro_f1": (
                    final_metrics[
                        "macro_f1"
                    ]
                ),

                "recalls": (
                    final_metrics[
                        "recalls"
                    ].tolist()
                ),

                "pred_counts": (
                    final_metrics[
                        "pred_counts"
                    ].tolist()
                ),

                "four_cm": (
                    final_metrics[
                        "four_cm"
                    ].tolist()
                ),

                "binary_cm": (
                    final_metrics[
                        "binary_cm"
                    ].tolist()
                ),
            },

            "auxiliary_metrics": {
                "binary_head_accuracy": (
                    final_evaluation[
                        "binary_head_accuracy"
                    ]
                ),

                "binary_head_cm": (
                    final_evaluation[
                        "binary_head_cm"
                    ].tolist()
                ),

                "abnormal_head_accuracy": (
                    final_evaluation[
                        "abnormal_head_accuracy"
                    ]
                ),

                "abnormal_head_recalls": (
                    final_evaluation[
                        "abnormal_head_recalls"
                    ].tolist()
                ),

                "abnormal_head_cm": (
                    final_evaluation[
                        "abnormal_head_cm"
                    ].tolist()
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