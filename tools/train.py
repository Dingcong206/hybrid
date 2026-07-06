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
        "checkpoints_d5_decoupled_hierarchical_seed42"
    ),

    # --------------------------------------------------------
    # 训练轮数
    # --------------------------------------------------------
    "EPOCHS": 50,

    # 三阶段训练
    "STAGE1_END": 10,
    "STAGE2_END": 35,

    # 官方训练集内部患者级验证比例
    "VAL_RATIO": 0.20,

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
    # 模型参数
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

    # --------------------------------------------------------
    # 手动轻度类别权重
    #
    # Four:
    # Normal / Crackle / Wheeze / Both
    # --------------------------------------------------------
    "FOUR_MANUAL_WEIGHTS": [
        1.00,
        1.00,
        1.15,
        1.35,
    ],

    # Abnormal:
    # Crackle / Wheeze / Both
    "ABNORMAL_MANUAL_WEIGHTS": [
        1.00,
        1.15,
        1.40,
    ],

    # --------------------------------------------------------
    # 损失函数
    # --------------------------------------------------------
    "FOUR_LABEL_SMOOTHING": 0.05,
    "ABNORMAL_LABEL_SMOOTHING": 0.05,

    # Binary Focal Loss
    "BINARY_FOCAL_GAMMA": 1.50,

    # Binary Head与Four Head聚合二分类概率的一致性约束
    "CONSISTENCY_WEIGHT": 0.10,

    # --------------------------------------------------------
    # SpecAugment
    # --------------------------------------------------------
    "USE_SPECAUGMENT": True,

    # Stage 1和Stage 2
    "STAGE12_TIME_MASK_MAX": 80,
    "STAGE12_FREQ_MASK_MAX": 16,

    # Stage 3降低增强
    "STAGE3_TIME_MASK_MAX": 40,
    "STAGE3_FREQ_MASK_MAX": 8,

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

    # Stage 3学习率缩小为原来的10%
    "STAGE3_LR_SCALE": 0.10,

    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # --------------------------------------------------------
    # 验证集二分类阈值搜索
    # --------------------------------------------------------
    "THRESHOLD_MIN": 0.30,
    "THRESHOLD_MAX": 0.70,
    "THRESHOLD_STEP": 0.01,

    # 异常类型预测时，Four Head所占权重
    "FOUR_SUBTYPE_WEIGHTS": [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ],

    # --------------------------------------------------------
    # 最佳模型选择指标
    #
    # Selection =
    # 0.50 * ICBHI Score
    # + 0.30 * Four-class Accuracy
    # + 0.20 * Binary Accuracy
    # --------------------------------------------------------
    "SELECTION_SCORE_WEIGHT": 0.50,
    "SELECTION_FOUR_ACC_WEIGHT": 0.30,
    "SELECTION_BINARY_ACC_WEIGHT": 0.20,

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
# 5. 保存CPU版State Dict
# ============================================================
def state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# ============================================================
# 6. 患者ID推断
# ============================================================
def infer_patient_id(path_value: str) -> str:
    """
    ICBHI文件一般类似：

        101_1b1_Al_sc_Meditron_xxx.npy

    第一个下划线前的数字即患者ID。
    """

    filename_stem = Path(str(path_value)).stem
    return filename_stem.split("_")[0]


def add_patient_groups(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    possible_patient_columns = [
        "patient_id",
        "patient",
        "patient_number",
        "subject_id",
        "subject",
    ]

    for column in possible_patient_columns:
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
# 7. 患者级内部训练/验证划分
# ============================================================
def patient_level_split(
    dataframe: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = add_patient_groups(dataframe)

    groups = dataframe["_patient_group"].to_numpy()

    labels = dataframe["label"].to_numpy(
        dtype=np.int64
    )

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        raise RuntimeError(
            "无法进行患者级划分：检测到的患者数量不足2。"
        )

    # 尝试不同随机种子，确保训练和验证均包含四类
    for offset in range(200):
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

        train_classes = np.unique(
            train_dataframe["label"]
        )

        validation_classes = np.unique(
            validation_dataframe["label"]
        )

        if (
            len(train_classes) == 4
            and len(validation_classes) == 4
        ):
            train_patients = set(
                train_dataframe["_patient_group"]
            )

            validation_patients = set(
                validation_dataframe["_patient_group"]
            )

            overlap = (
                train_patients
                & validation_patients
            )

            if overlap:
                raise RuntimeError(
                    "患者级划分失败，训练集和验证集存在患者重叠。"
                )

            return (
                train_dataframe,
                validation_dataframe,
            )

    raise RuntimeError(
        "多次尝试后仍无法得到同时包含四类的患者级验证集。"
    )


# ============================================================
# 8. 三阶段训练配置
# ============================================================
def get_stage_config(
    epoch: int,
    cfg,
) -> Dict[str, object]:
    # --------------------------------------------------------
    # Stage 1：优先强化Normal/Abnormal区分
    # --------------------------------------------------------
    if epoch <= int(cfg["STAGE1_END"]):
        return {
            "name": "Stage-1 Binary Warmup",

            "four_loss_weight": 0.50,
            "binary_loss_weight": 1.00,
            "abnormal_loss_weight": 0.25,

            "consistency_weight": float(
                cfg["CONSISTENCY_WEIGHT"]
            ),

            "time_mask_max": int(
                cfg["STAGE12_TIME_MASK_MAX"]
            ),

            "freq_mask_max": int(
                cfg["STAGE12_FREQ_MASK_MAX"]
            ),

            "lr_scale": 1.00,
        }

    # --------------------------------------------------------
    # Stage 2：联合优化四分类和异常类型
    # --------------------------------------------------------
    if epoch <= int(cfg["STAGE2_END"]):
        return {
            "name": "Stage-2 Joint Training",

            "four_loss_weight": 1.00,
            "binary_loss_weight": 0.50,
            "abnormal_loss_weight": 0.80,

            "consistency_weight": float(
                cfg["CONSISTENCY_WEIGHT"]
            ),

            "time_mask_max": int(
                cfg["STAGE12_TIME_MASK_MAX"]
            ),

            "freq_mask_max": int(
                cfg["STAGE12_FREQ_MASK_MAX"]
            ),

            "lr_scale": 1.00,
        }

    # --------------------------------------------------------
    # Stage 3：低学习率、弱增强微调
    # --------------------------------------------------------
    return {
        "name": "Stage-3 Low-LR Fine-tuning",

        "four_loss_weight": 1.00,
        "binary_loss_weight": 0.50,
        "abnormal_loss_weight": 0.80,

        "consistency_weight": float(
            cfg["CONSISTENCY_WEIGHT"]
        ),

        "time_mask_max": int(
            cfg["STAGE3_TIME_MASK_MAX"]
        ),

        "freq_mask_max": int(
            cfg["STAGE3_FREQ_MASK_MAX"]
        ),

        "lr_scale": float(
            cfg["STAGE3_LR_SCALE"]
        ),
    }


# ============================================================
# 9. Warmup + Cosine学习率
# ============================================================
def set_epoch_lrs(
    optimizer,
    base_lrs,
    minimum_lrs,
    epoch: int,
    schedule_total_epochs: int,
    warmup_epochs: int,
    stage_lr_scale: float,
):
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

    current_lrs = [
        learning_rate
        * stage_lr_scale
        for learning_rate in current_lrs
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
# 10. SpecAugment
# ============================================================
def apply_specaugment(
    fbank: torch.Tensor,
    time_mask_max: int,
    frequency_mask_max: int,
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

    time_frames = int(x.shape[0])
    frequency_bins = int(x.shape[1])

    mask_value = x.mean()

    # Time Mask
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
                time_frames - time_width,
            )

            x[
                time_start:
                time_start + time_width,
                :
            ] = mask_value

    # Frequency Mask
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
# 11. Dataset
# ============================================================
class FbankDataset(Dataset):
    """
    DataFrame必须包含：

        fbank_path
        label

    单个Fbank：
        [798,128]

    返回：
        x = [1,798,128]
        y = label
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

        self.time_mask_max = int(
            cfg["STAGE12_TIME_MASK_MAX"]
        )

        self.frequency_mask_max = int(
            cfg["STAGE12_FREQ_MASK_MAX"]
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
                "数据表缺少必要列："
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

    def set_augmentation(
        self,
        time_mask_max: int,
        frequency_mask_max: int,
    ) -> None:
        self.time_mask_max = int(
            time_mask_max
        )

        self.frequency_mask_max = int(
            frequency_mask_max
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

    def __getitem__(self, index):
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
                time_mask_max=(
                    self.time_mask_max
                ),
                frequency_mask_max=(
                    self.frequency_mask_max
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
# 12. DataLoader
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

        # 每轮重新创建worker，
        # 保证阶段增强参数更新后能够生效
        "persistent_workers": False,
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
# 13. 构建模型
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

        d_state=int(
            cfg["D_STATE"]
        ),

        d_conv=int(
            cfg["D_CONV"]
        ),

        expand=int(
            cfg["EXPAND"]
        ),
    ).to(device)

    return model


# ============================================================
# 14. 优化器
# ============================================================
def build_optimizer(
    model,
    cfg,
):
    task_parameters = (
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
                "params": task_parameters,
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
# 15. 类别权重
# ============================================================
def build_loss_weights(
    cfg,
    device,
):
    four_weight = torch.tensor(
        cfg["FOUR_MANUAL_WEIGHTS"],
        dtype=torch.float32,
        device=device,
    )

    abnormal_weight = torch.tensor(
        cfg["ABNORMAL_MANUAL_WEIGHTS"],
        dtype=torch.float32,
        device=device,
    )

    if four_weight.numel() != 4:
        raise ValueError(
            "FOUR_MANUAL_WEIGHTS必须包含4个值。"
        )

    if abnormal_weight.numel() != 3:
        raise ValueError(
            "ABNORMAL_MANUAL_WEIGHTS必须包含3个值。"
        )

    print(
        "[Loss] Four weights:",
        four_weight.detach().cpu().tolist(),
    )

    print(
        "[Loss] Abnormal weights:",
        abnormal_weight.detach().cpu().tolist(),
    )

    return {
        "four": four_weight,
        "abnormal": abnormal_weight,
    }


# ============================================================
# 16. Binary Focal Loss
# ============================================================
def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    cross_entropy = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    probability = torch.softmax(
        logits,
        dim=1,
    )

    target_probability = probability.gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1)

    focal_weight = (
        1.0
        - target_probability
    ).pow(gamma)

    return (
        focal_weight
        * cross_entropy
    ).mean()


# ============================================================
# 17. 多任务损失
# ============================================================
def calculate_multitask_loss(
    outputs,
    labels,
    loss_weights,
    cfg,
    stage,
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
    # Binary Focal Loss
    # Normal=0，Abnormal=1
    # --------------------------------------------------------
    binary_labels = (
        labels > 0
    ).long()

    binary_loss = focal_cross_entropy(
        logits=outputs["binary_logits"],
        targets=binary_labels,
        gamma=float(
            cfg["BINARY_FOCAL_GAMMA"]
        ),
    )

    # --------------------------------------------------------
    # Abnormal Subtype Loss
    #
    # 原始：
    # 1=Crackle，2=Wheeze，3=Both
    #
    # 子任务：
    # 0=Crackle，1=Wheeze，2=Both
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
            outputs["abnormal_logits"].sum()
            * 0.0
        )

    # --------------------------------------------------------
    # Binary/Four一致性损失
    #
    # Four Head被detach，
    # 避免Binary任务反向干扰Four Head。
    # --------------------------------------------------------
    probabilities = (
        DTFHybridModel
        .build_probabilities(outputs)
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
            stage["four_loss_weight"]
        )
        * four_loss

        + float(
            stage["binary_loss_weight"]
        )
        * binary_loss

        + float(
            stage["abnormal_loss_weight"]
        )
        * abnormal_loss

        + float(
            stage["consistency_weight"]
        )
        * consistency_loss
    )

    return {
        "total_loss": total_loss,
        "four_loss": four_loss,
        "binary_loss": binary_loss,
        "abnormal_loss": abnormal_loss,
        "consistency_loss": consistency_loss,
        "abnormal_count": abnormal_count,
    }


# ============================================================
# 18. 单轮训练
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
    stage,
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

        # 计算当前梯度累积窗口的实际batch数量
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
                loss_weights=loss_weights,
                cfg=cfg,
                stage=stage,
            )

            backward_loss = (
                loss_result["total_loss"]
                / current_accumulation_size
            )

        scaler.scale(
            backward_loss
        ).backward()

        batch_size = int(y.shape[0])

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

            scaler.step(optimizer)
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
            / max(
                total_abnormal_samples,
                1,
            )
        ),

        "consistency_loss": (
            accumulated_consistency_loss
            / max(total_samples, 1)
        ),
    }


# ============================================================
# 19. 收集模型概率
# ============================================================
@torch.no_grad()
def collect_probabilities(
    loader,
    model,
    device,
    use_amp: bool,
):
    model.eval()

    all_labels = []
    all_four_probabilities = []
    all_binary_probabilities = []
    all_abnormal_probabilities = []

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
                    outputs
                )
            )

        all_labels.append(
            y.cpu()
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

    return {
        "labels": torch.cat(
            all_labels
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
    }


# ============================================================
# 20. Numpy硬层级推理
# ============================================================
def hard_predict_numpy(
    probabilities,
    threshold: float,
    four_subtype_weight: float,
):
    four_probability = probabilities[
        "four_probability"
    ]

    binary_probability = probabilities[
        "binary_probability"
    ]

    abnormal_probability = probabilities[
        "abnormal_probability"
    ]

    four_subtype_probability = (
        four_probability[:, 1:]
    )

    four_subtype_probability = (
        four_subtype_probability
        / np.clip(
            four_subtype_probability.sum(
                axis=1,
                keepdims=True,
            ),
            1e-8,
            None,
        )
    )

    subtype_probability = (
        four_subtype_weight
        * four_subtype_probability

        + (
            1.0
            - four_subtype_weight
        )
        * abnormal_probability
    )

    subtype_prediction = (
        np.argmax(
            subtype_probability,
            axis=1,
        )
        + 1
    )

    prediction = np.where(
        binary_probability[:, 1]
        >= threshold,

        subtype_prediction,

        0,
    ).astype(np.int64)

    return prediction


# ============================================================
# 21. 指标计算
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
            binary_pred,
            labels=[0, 1],
        ),
    }


# ============================================================
# 22. 模型选择指标
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
                "SELECTION_FOUR_ACC_WEIGHT"
            ]
        )
        * metrics["accuracy"]

        + float(
            cfg[
                "SELECTION_BINARY_ACC_WEIGHT"
            ]
        )
        * metrics["binary_accuracy"]
    )


# ============================================================
# 23. 搜索阈值和异常融合权重
# ============================================================
def search_decision_parameters(
    probabilities,
    cfg,
):
    y_true = probabilities["labels"]

    thresholds = np.arange(
        float(
            cfg["THRESHOLD_MIN"]
        ),

        float(
            cfg["THRESHOLD_MAX"]
        )
        + 1e-9,

        float(
            cfg["THRESHOLD_STEP"]
        ),
    )

    subtype_weights = [
        float(value)
        for value in cfg[
            "FOUR_SUBTYPE_WEIGHTS"
        ]
    ]

    best_result = None

    for threshold in thresholds:
        for subtype_weight in subtype_weights:
            prediction = hard_predict_numpy(
                probabilities=probabilities,

                threshold=float(
                    threshold
                ),

                four_subtype_weight=(
                    subtype_weight
                ),
            )

            metrics = calculate_metrics(
                y_true,
                prediction,
            )

            selection_value = (
                calculate_selection_value(
                    metrics,
                    cfg,
                )
            )

            current_result = {
                "threshold": float(
                    threshold
                ),

                "four_subtype_weight": float(
                    subtype_weight
                ),

                "selection_value": float(
                    selection_value
                ),

                "metrics": metrics,
            }

            if best_result is None:
                best_result = current_result
                continue

            current_key = (
                current_result[
                    "selection_value"
                ],

                current_result[
                    "metrics"
                ]["macro_f1"],

                current_result[
                    "metrics"
                ]["score"],
            )

            best_key = (
                best_result[
                    "selection_value"
                ],

                best_result[
                    "metrics"
                ]["macro_f1"],

                best_result[
                    "metrics"
                ]["score"],
            )

            if current_key > best_key:
                best_result = current_result

    return best_result


# ============================================================
# 24. 辅助分类头指标
# ============================================================
def calculate_auxiliary_metrics(
    probabilities,
):
    y_true = probabilities["labels"]

    binary_true = (
        y_true > 0
    ).astype(np.int64)

    binary_prediction = np.argmax(
        probabilities[
            "binary_probability"
        ],
        axis=1,
    )

    abnormal_mask = (
        y_true > 0
    )

    abnormal_true = (
        y_true[abnormal_mask]
        - 1
    )

    abnormal_prediction = np.argmax(
        probabilities[
            "abnormal_probability"
        ][abnormal_mask],
        axis=1,
    )

    return {
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

        "abnormal_head_cm": confusion_matrix(
            abnormal_true,
            abnormal_prediction,
            labels=[0, 1, 2],
        ),

        "abnormal_head_recalls": recall_score(
            abnormal_true,
            abnormal_prediction,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        ),
    }


# ============================================================
# 25. 形状测试
# ============================================================
@torch.no_grad()
def shape_test(
    loader,
    model,
    device,
) -> None:
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

    probabilities = (
        model.build_probabilities(
            outputs
        )
    )

    prediction_result = (
        model.hard_hierarchical_predict(
            probabilities,
            binary_threshold=0.5,
            four_subtype_weight=0.30,
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
        "[Shape] Shared Feature:",
        tuple(
            outputs[
                "shared_feature"
            ].shape
        ),
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
        "[Shape] Prediction:",
        tuple(
            prediction_result[
                "prediction"
            ].shape
        ),
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
    ) == (4,)

    assert tuple(
        outputs[
            "binary_logits"
        ].shape[1:]
    ) == (2,)

    assert tuple(
        outputs[
            "abnormal_logits"
        ].shape[1:]
    ) == (3,)

    print(
        "[PASS] D5模型连接成功。",
        flush=True,
    )


# ============================================================
# 26. 打印指标
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


# ============================================================
# 27. 通用训练流程
# ============================================================
def run_training(
    model,
    train_loader,
    train_dataset,
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
        stage = get_stage_config(
            epoch,
            cfg,
        )

        train_dataset.set_augmentation(
            time_mask_max=(
                stage[
                    "time_mask_max"
                ]
            ),

            frequency_mask_max=(
                stage[
                    "freq_mask_max"
                ]
            ),
        )

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

                # 全量重训练时仍使用与开发阶段相同的
                # 50轮学习率轨迹，保证第N轮配置一致
                schedule_total_epochs=int(
                    cfg["EPOCHS"]
                ),

                warmup_epochs=int(
                    cfg["WARMUP_EPOCHS"]
                ),

                stage_lr_scale=float(
                    stage["lr_scale"]
                ),
            )
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
            stage=stage,
        )

        elapsed_time = (
            time.time()
            - epoch_start_time
        )

        history_row = {
            "epoch": epoch,
            "stage": stage["name"],

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
        # 内部验证
        # --------------------------------------------------------
        if validation_loader is not None:
            validation_probabilities = (
                collect_probabilities(
                    loader=validation_loader,
                    model=model,
                    device=device,
                    use_amp=use_amp,
                )
            )

            decision_result = (
                search_decision_parameters(
                    probabilities=(
                        validation_probabilities
                    ),
                    cfg=cfg,
                )
            )

            validation_metrics = (
                decision_result["metrics"]
            )

            history_row.update(
                {
                    "val_selection": (
                        decision_result[
                            "selection_value"
                        ]
                    ),

                    "val_threshold": (
                        decision_result[
                            "threshold"
                        ]
                    ),

                    "val_four_subtype_weight": (
                        decision_result[
                            "four_subtype_weight"
                        ]
                    ),

                    "val_score": (
                        validation_metrics[
                            "score"
                        ]
                    ),

                    "val_four_accuracy": (
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
                }
            )

            is_best = False

            if best_result is None:
                is_best = True
            else:
                current_key = (
                    decision_result[
                        "selection_value"
                    ],
                    validation_metrics[
                        "macro_f1"
                    ],
                    validation_metrics[
                        "score"
                    ],
                )

                best_key = (
                    best_result[
                        "selection_value"
                    ],
                    best_result[
                        "metrics"
                    ]["macro_f1"],
                    best_result[
                        "metrics"
                    ]["score"],
                )

                if current_key > best_key:
                    is_best = True

            if is_best:
                best_result = {
                    "epoch": epoch,

                    "selection_value": float(
                        decision_result[
                            "selection_value"
                        ]
                    ),

                    "threshold": float(
                        decision_result[
                            "threshold"
                        ]
                    ),

                    "four_subtype_weight": float(
                        decision_result[
                            "four_subtype_weight"
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

                if best_checkpoint_path is not None:
                    torch.save(
                        {
                            "epoch": (
                                best_result[
                                    "epoch"
                                ]
                            ),

                            "model_state": (
                                best_result[
                                    "model_state"
                                ]
                            ),

                            "threshold": (
                                best_result[
                                    "threshold"
                                ]
                            ),

                            "four_subtype_weight": (
                                best_result[
                                    "four_subtype_weight"
                                ]
                            ),

                            "selection_value": (
                                best_result[
                                    "selection_value"
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

            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{total_epochs} | "

                f"{stage['name']} | "

                f"Loss "
                f"{train_result['total_loss']:.4f} | "

                f"ValScore "
                f"{validation_metrics['score']:.2f} | "

                f"FourAcc "
                f"{validation_metrics['accuracy']:.2f} | "

                f"BinAcc "
                f"{validation_metrics['binary_accuracy']:.2f} | "

                f"MacroF1 "
                f"{validation_metrics['macro_f1']:.4f} | "

                f"Thr "
                f"{decision_result['threshold']:.2f} | "

                f"SubW "
                f"{decision_result['four_subtype_weight']:.2f} | "

                f"BestEpoch "
                f"{best_result['epoch']} | "

                f"{elapsed_time:.1f}s",
                flush=True,
            )

        else:
            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{total_epochs} | "

                f"{stage['name']} | "

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

    return best_result, history


# ============================================================
# 28. 主函数
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
        "[Protocol] 官方训练集先进行患者级内部验证。"
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
    # 第一阶段：内部验证开发
    # ========================================================
    internal_train_dataset = FbankDataset(
        dataframe=(
            internal_train_dataframe
        ),
        base_directory=root,
        cfg=cfg,
        training=True,
    )

    internal_validation_dataset = (
        FbankDataset(
            dataframe=(
                internal_validation_dataframe
            ),
            base_directory=root,
            cfg=cfg,
            training=False,
        )
    )

    internal_train_loader = make_loader(
        dataset=internal_train_dataset,
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
        base_learning_rates,
        minimum_learning_rates,
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
        "D5 DEVELOPMENT: "
        "PATIENT-LEVEL INTERNAL VALIDATION"
    )
    print("=" * 100)

    best_result, _ = run_training(
        model=development_model,

        train_loader=(
            internal_train_loader
        ),

        train_dataset=(
            internal_train_dataset
        ),

        optimizer=(
            development_optimizer
        ),

        base_learning_rates=(
            base_learning_rates
        ),

        minimum_learning_rates=(
            minimum_learning_rates
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

    if best_result is None:
        raise RuntimeError(
            "内部验证阶段未产生最佳模型。"
        )

    print_metrics(
        title="BEST INTERNAL VALIDATION",

        metrics=best_result[
            "metrics"
        ],
    )

    print(
        f"Best epoch: "
        f"{best_result['epoch']}"
    )

    print(
        f"Best binary threshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Best four subtype weight: "
        f"{best_result['four_subtype_weight']:.2f}"
    )

    # 删除开发模型，释放显存
    del development_model
    del development_optimizer

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ========================================================
    # 第二阶段：完整官方训练集重新训练
    # ========================================================
    print()
    print("=" * 100)
    print(
        "D5 FINAL RETRAIN: "
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
    )

    (
        final_optimizer,
        final_base_learning_rates,
        final_minimum_learning_rates,
    ) = build_optimizer(
        final_model,
        cfg,
    )

    final_loss_weights = build_loss_weights(
        cfg,
        device,
    )

    run_training(
        model=final_model,

        train_loader=full_train_loader,

        train_dataset=full_train_dataset,

        optimizer=final_optimizer,

        base_learning_rates=(
            final_base_learning_rates
        ),

        minimum_learning_rates=(
            final_minimum_learning_rates
        ),

        loss_weights=(
            final_loss_weights
        ),

        device=device,

        use_amp=use_amp,

        cfg=cfg,

        # 使用内部验证选出的最佳轮数
        total_epochs=int(
            best_result["epoch"]
        ),

        validation_loader=None,

        best_checkpoint_path=None,

        history_path=(
            full_train_history_path
        ),
    )

    # ========================================================
    # 第三阶段：官方测试集最终评估
    # ========================================================
    test_probabilities = collect_probabilities(
        loader=official_test_loader,
        model=final_model,
        device=device,
        use_amp=use_amp,
    )

    test_prediction = hard_predict_numpy(
        probabilities=test_probabilities,

        threshold=float(
            best_result["threshold"]
        ),

        four_subtype_weight=float(
            best_result[
                "four_subtype_weight"
            ]
        ),
    )

    final_metrics = calculate_metrics(
        y_true=test_probabilities[
            "labels"
        ],

        y_pred=test_prediction,
    )

    auxiliary_result = (
        calculate_auxiliary_metrics(
            test_probabilities
        )
    )

    print()
    print("=" * 100)
    print("FINAL OFFICIAL TEST RESULT")
    print("=" * 100)

    print_metrics(
        title=(
            "D5 HARD HIERARCHICAL PREDICTION"
        ),
        metrics=final_metrics,
    )

    print()
    print("-" * 80)
    print("AUXILIARY HEAD RESULTS")
    print("-" * 80)

    print(
        f"Binary Head Accuracy: "
        f"{auxiliary_result['binary_head_accuracy']:.4f}"
    )

    print(
        "Binary Head Confusion Matrix:"
    )

    print(
        auxiliary_result[
            "binary_head_cm"
        ]
    )

    print()

    print(
        f"Abnormal Head Accuracy: "
        f"{auxiliary_result['abnormal_head_accuracy']:.4f}"
    )

    print(
        "Abnormal Head Recall "
        "[Crackle, Wheeze, Both]:",
        np.round(
            auxiliary_result[
                "abnormal_head_recalls"
            ],
            4,
        ).tolist(),
    )

    print(
        "Abnormal Head Confusion Matrix:"
    )

    print(
        auxiliary_result[
            "abnormal_head_cm"
        ]
    )

    print()

    print(
        f"Selected epoch: "
        f"{best_result['epoch']}"
    )

    print(
        f"Selected binary threshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Selected four subtype weight: "
        f"{best_result['four_subtype_weight']:.2f}"
    )

    # ========================================================
    # 保存最终模型
    # ========================================================
    torch.save(
        {
            "epoch": int(
                best_result["epoch"]
            ),

            "model_state": (
                state_dict_to_cpu(
                    final_model
                )
            ),

            "config": deepcopy(cfg),

            "binary_threshold": float(
                best_result["threshold"]
            ),

            "four_subtype_weight": float(
                best_result[
                    "four_subtype_weight"
                ]
            ),

            "internal_validation_metrics": {
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
                    final_metrics["score"]
                ),

                "sp": (
                    final_metrics["sp"]
                ),

                "se": (
                    final_metrics["se"]
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
        "Full-train history:",
        full_train_history_path,
    )

    print(
        "Final model:",
        final_model_path,
    )


if __name__ == "__main__":
    main()