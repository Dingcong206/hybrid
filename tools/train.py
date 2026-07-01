#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# 1. 配置
# ============================================================
CONFIG = {
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_ast_patch_tokens"
    ),

    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_direct4_consistent_v1"
    ),

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    "BATCH_SIZE": 4,

    # Effective batch size = 4 × 8 = 32
    "ACCUM_STEPS": 8,

    "NUM_WORKERS": 1,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    "EPOCHS": 50,

    # 前 20 轮不早停
    "MIN_EPOCHS": 20,

    # 20 轮以后连续 12 轮无提升才停止
    "PATIENCE": 12,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------
    "INPUT_DIM": 768,
    "D_MODEL": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,
    "DROPOUT": 0.15,

    # --------------------------------------------------------
    # Head
    # --------------------------------------------------------
    "HEAD_DROPOUT": 0.20,

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------
    "BACKBONE_LR": 1e-5,
    "HEAD_LR": 5e-5,

    "MIN_BACKBONE_LR": 1e-6,
    "MIN_HEAD_LR": 5e-6,

    "WARMUP_EPOCHS": 3,

    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # --------------------------------------------------------
    # Loss
    #
    # 四分类、二分类和异常子类分类
    # 全部由同一组 four-class logits 计算。
    # --------------------------------------------------------
    "FOUR_LOSS_WEIGHT": 1.0,
    "BINARY_LOSS_WEIGHT": 0.35,
    "SUBTYPE_LOSS_WEIGHT": 0.40,

    "LABEL_SMOOTHING": 0.02,

    # --------------------------------------------------------
    # 轻度类别权重
    # --------------------------------------------------------
    "FOUR_WEIGHT_POWER": 0.25,
    "FOUR_WEIGHT_MAX": 1.50,

    "SUBTYPE_WEIGHT_POWER": 0.25,
    "SUBTYPE_WEIGHT_MAX": 1.40,

    # --------------------------------------------------------
    # 验证集异常偏置搜索
    #
    # 给类别 1、2、3 的 logits 同时增加一个 bias，
    # 只调节 Normal / Abnormal 平衡，
    # 不改变异常三类之间的排序。
    # --------------------------------------------------------
    "BIAS_MIN": -1.50,
    "BIAS_MAX": 1.50,
    "BIAS_STEP": 0.05,

    # 避免全 Normal 或全 Abnormal 模型被选中
    "MIN_VALID_SP": 30.0,
    "MIN_VALID_SE": 30.0,
}


# ============================================================
# 2. 导入项目模型
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
    TimeFrequencyEncoder,
)


# ============================================================
# 3. 随机种子
# ============================================================
def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )

    if torch.cuda.is_available():

        torch.backends.cuda.matmul.allow_tf32 = (
            True
        )

        torch.backends.cudnn.allow_tf32 = (
            True
        )

    try:

        torch.set_float32_matmul_precision(
            "high"
        )

    except AttributeError:

        pass


# ============================================================
# 4. AMP
# ============================================================
def make_scaler(
    enabled,
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
            enabled=enabled
        )


# ============================================================
# 5. 加载 checkpoint
# ============================================================
def safe_load(
    path,
    device,
):

    try:

        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location=device,
        )


# ============================================================
# 6. Warm-up + Cosine LR
# ============================================================
def set_lrs(
    optimizer,
    base_lrs,
    min_lrs,
    epoch,
    total_epochs,
    warmup_epochs,
):

    if epoch <= warmup_epochs:

        scale = (
            0.20
            +
            0.80
            * epoch
            / max(
                warmup_epochs,
                1,
            )
        )

        current_lrs = [
            lr * scale
            for lr in base_lrs
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

        ratio = (
            0.5
            * (
                1.0
                +
                math.cos(
                    math.pi
                    * cosine_step
                    / cosine_total
                )
            )
        )

        current_lrs = [
            min_lr
            +
            (
                base_lr
                - min_lr
            )
            * ratio

            for (
                base_lr,
                min_lr,
            )
            in zip(
                base_lrs,
                min_lrs,
            )
        ]

    for (
        parameter_group,
        current_lr,
    ) in zip(
        optimizer.param_groups,
        current_lrs,
    ):

        parameter_group[
            "lr"
        ] = float(
            current_lr
        )

    return current_lrs


# ============================================================
# 7. Backbone
# ============================================================
def make_backbone(
    cfg,
):

    return TimeFrequencyEncoder(
        input_dim=(
            cfg[
                "INPUT_DIM"
            ]
        ),

        d_model=(
            cfg[
                "D_MODEL"
            ]
        ),

        freq_patches=(
            cfg[
                "FREQ_PATCHES"
            ]
        ),

        time_patches=(
            cfg[
                "TIME_PATCHES"
            ]
        ),

        time_depth=(
            cfg[
                "TIME_DEPTH"
            ]
        ),

        freq_depth=(
            cfg[
                "FREQ_DEPTH"
            ]
        ),

        num_heads=(
            cfg[
                "NHEAD"
            ]
        ),

        dropout=(
            cfg[
                "DROPOUT"
            ]
        ),
    )


# ============================================================
# 8. 单头直接四分类模型
# ============================================================
class DirectFourClassModel(
    nn.Module
):

    def __init__(
        self,
        cfg,
    ):

        super().__init__()

        self.backbone = make_backbone(
            cfg
        )

        self.head = nn.Sequential(
            nn.LayerNorm(
                cfg[
                    "D_MODEL"
                ]
            ),

            nn.Dropout(
                cfg[
                    "HEAD_DROPOUT"
                ]
            ),

            nn.Linear(
                cfg[
                    "D_MODEL"
                ],
                4,
            ),
        )

    def forward(
        self,
        x,
    ):

        feature = self.backbone(
            x
        )

        logits = self.head(
            feature
        )

        return logits


# ============================================================
# 9. Dataset
# ============================================================
class TokenDataset(
    Dataset
):

    def __init__(
        self,
        csv_path,
        cfg,
    ):

        super().__init__()

        self.csv_path = Path(
            csv_path
        )

        self.df = pd.read_csv(
            self.csv_path
        ).reset_index(
            drop=True
        )

        required_columns = {
            "tokens_path",
            "label",
        }

        if not required_columns.issubset(
            self.df.columns
        ):

            raise ValueError(
                f"{csv_path} 必须包含 "
                f"tokens_path 和 label"
            )

        self.df[
            "label"
        ] = (
            self.df[
                "label"
            ]
            .astype(int)
        )

        self.labels = (
            self.df[
                "label"
            ]
            .to_numpy(
                dtype=np.int64
            )
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
                "label 必须为 0、1、2、3，"
                f"发现："
                f"{invalid_labels.tolist()}"
            )

        self.expected_shape = (
            cfg[
                "FREQ_PATCHES"
            ]
            *
            cfg[
                "TIME_PATCHES"
            ],

            cfg[
                "INPUT_DIM"
            ],
        )

        self.class_counts = (
            np.bincount(
                self.labels,
                minlength=4,
            )
        )

        print(
            f"[Dataset] "
            f"samples="
            f"{len(self.df)} | "
            f"counts="
            f"{self.class_counts.tolist()} | "
            f"{csv_path}"
        )

    def __len__(
        self,
    ):

        return len(
            self.df
        )

    def resolve_path(
        self,
        raw_path,
    ):

        token_path = Path(
            raw_path
        )

        if token_path.exists():

            return token_path

        relative_path = (
            self.csv_path.parent
            / token_path
        )

        if relative_path.exists():

            return relative_path

        raise FileNotFoundError(
            f"Token 文件不存在："
            f"{raw_path}"
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        token_path = self.resolve_path(
            str(
                row[
                    "tokens_path"
                ]
            )
        )

        tokens = np.load(
            token_path
        )

        if tuple(
            tokens.shape
        ) != self.expected_shape:

            raise ValueError(
                f"{token_path} "
                f"shape="
                f"{tuple(tokens.shape)}，"
                f"expected="
                f"{self.expected_shape}"
            )

        x = torch.from_numpy(
            tokens
        ).float()

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
# 10. DataLoader
# ============================================================
def collate_fixed(
    batch,
):

    xs, ys = zip(
        *batch
    )

    return (
        torch.stack(
            xs
        ),

        torch.stack(
            ys
        ).view(-1),
    )


def make_loader(
    dataset,
    cfg,
    device,
    shuffle,
):

    workers = cfg[
        "NUM_WORKERS"
    ]

    return DataLoader(
        dataset,

        batch_size=(
            cfg[
                "BATCH_SIZE"
            ]
        ),

        shuffle=shuffle,

        num_workers=workers,

        pin_memory=(
            device.type
            == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        drop_last=False,

        collate_fn=collate_fixed,
    )


# ============================================================
# 11. 类别权重
# ============================================================
def build_weights(
    class_counts,
    power,
    maximum_weight,
    abnormal_only=False,
):

    if abnormal_only:

        counts = np.asarray(
            class_counts[
                1:4
            ],
            dtype=np.float64,
        )

        reference_count = (
            counts.max()
        )

    else:

        counts = np.asarray(
            class_counts,
            dtype=np.float64,
        )

        reference_count = (
            counts[0]
        )

    weights = np.power(
        reference_count
        /
        np.maximum(
            counts,
            1.0,
        ),

        power,
    )

    if not abnormal_only:

        weights[0] = 1.0

    weights = np.clip(
        weights,
        1.0,
        maximum_weight,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# 12. 一致性层级损失
#
# 只有一组四分类 logits。
#
# binary_logits:
#   Normal = logit[0]
#   Abnormal = logsumexp(logit[1:4])
#
# subtype_logits:
#   logit[1:4]
# ============================================================
def calculate_loss(
    logits,
    labels,
    four_class_weights,
    subtype_weights,
    cfg,
):

    four_loss = F.cross_entropy(
        logits,
        labels,

        weight=(
            four_class_weights
        ),

        label_smoothing=(
            cfg[
                "LABEL_SMOOTHING"
            ]
        ),
    )

    binary_logits = torch.stack(
        [
            logits[
                :,
                0,
            ],

            torch.logsumexp(
                logits[
                    :,
                    1:4,
                ],
                dim=1,
            ),
        ],

        dim=1,
    )

    binary_target = (
        labels > 0
    ).long()

    binary_loss = F.cross_entropy(
        binary_logits,
        binary_target,
    )

    abnormal_mask = (
        labels > 0
    )

    if abnormal_mask.any():

        subtype_logits = logits[
            abnormal_mask,
            1:4,
        ]

        subtype_target = (
            labels[
                abnormal_mask
            ]
            - 1
        )

        subtype_loss = F.cross_entropy(
            subtype_logits,
            subtype_target,

            weight=(
                subtype_weights
            ),

            label_smoothing=(
                cfg[
                    "LABEL_SMOOTHING"
                ]
            ),
        )

    else:

        subtype_loss = torch.zeros(
            (),
            dtype=logits.dtype,
            device=logits.device,
        )

    total_loss = (
        cfg[
            "FOUR_LOSS_WEIGHT"
        ]
        * four_loss

        +

        cfg[
            "BINARY_LOSS_WEIGHT"
        ]
        * binary_loss

        +

        cfg[
            "SUBTYPE_LOSS_WEIGHT"
        ]
        * subtype_loss
    )

    return (
        total_loss,
        four_loss,
        binary_loss,
        subtype_loss,
    )


# ============================================================
# 13. 指标
# ============================================================
def calculate_metrics(
    y_true,
    y_pred,
):

    cm = confusion_matrix(
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
            cm[0].sum()
        ),
        1,
    )

    abnormal_total = max(
        int(
            cm[1:].sum()
        ),
        1,
    )

    specificity = (
        100.0
        * float(
            cm[
                0,
                0,
            ]
        )
        / normal_total
    )

    sensitivity = (
        100.0
        * float(
            cm[
                1,
                1,
            ]
            +
            cm[
                2,
                2,
            ]
            +
            cm[
                3,
                3,
            ]
        )
        / abnormal_total
    )

    score = (
        specificity
        +
        sensitivity
    ) / 2.0

    return {
        "FOUR_SCORE": float(
            score
        ),

        "FOUR_SP": float(
            specificity
        ),

        "FOUR_SE": float(
            sensitivity
        ),

        "ACC": float(
            accuracy_score(
                y_true,
                y_pred,
            )
            * 100.0
        ),

        "MACRO_F1": float(
            f1_score(
                y_true,
                y_pred,

                average="macro",

                zero_division=0,
            )
        ),

        "RECALL": recall_score(
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

        "PRED_COUNTS": (
            np.bincount(
                y_pred,
                minlength=4,
            )
        ),

        "FOUR_CM": cm,

        "BINARY_CM": confusion_matrix(
            (
                y_true > 0
            ).astype(
                np.int64
            ),

            (
                y_pred > 0
            ).astype(
                np.int64
            ),

            labels=[
                0,
                1,
            ],
        ),
    }


# ============================================================
# 14. 异常偏置
# ============================================================
def apply_abnormal_bias(
    logits,
    abnormal_bias,
):

    adjusted_logits = (
        logits.copy()
    )

    adjusted_logits[
        :,
        1:4,
    ] += float(
        abnormal_bias
    )

    return adjusted_logits


def search_abnormal_bias(
    logits,
    labels,
    cfg,
):

    best_result = None
    fallback_result = None

    bias_values = np.arange(
        cfg[
            "BIAS_MIN"
        ],

        cfg[
            "BIAS_MAX"
        ]
        +
        cfg[
            "BIAS_STEP"
        ]
        / 2.0,

        cfg[
            "BIAS_STEP"
        ],
    )

    for abnormal_bias in bias_values:

        prediction = np.argmax(
            apply_abnormal_bias(
                logits,
                abnormal_bias,
            ),

            axis=1,
        )

        current_result = {
            **calculate_metrics(
                labels,
                prediction,
            ),

            "ABNORMAL_BIAS": float(
                abnormal_bias
            ),
        }

        if (
            fallback_result is None
            or
            current_result[
                "MACRO_F1"
            ]
            >
            fallback_result[
                "MACRO_F1"
            ]
            or
            (
                current_result[
                    "MACRO_F1"
                ]
                ==
                fallback_result[
                    "MACRO_F1"
                ]
                and
                current_result[
                    "FOUR_SCORE"
                ]
                >
                fallback_result[
                    "FOUR_SCORE"
                ]
            )
        ):

            fallback_result = (
                current_result
            )

        valid_candidate = (
            current_result[
                "FOUR_SP"
            ]
            >=
            cfg[
                "MIN_VALID_SP"
            ]
            and
            current_result[
                "FOUR_SE"
            ]
            >=
            cfg[
                "MIN_VALID_SE"
            ]
        )

        if not valid_candidate:

            continue

        if (
            best_result is None
            or
            current_result[
                "FOUR_SCORE"
            ]
            >
            best_result[
                "FOUR_SCORE"
            ]
            or
            (
                current_result[
                    "FOUR_SCORE"
                ]
                ==
                best_result[
                    "FOUR_SCORE"
                ]
                and
                current_result[
                    "MACRO_F1"
                ]
                >
                best_result[
                    "MACRO_F1"
                ]
            )
        ):

            best_result = (
                current_result
            )

    if best_result is not None:

        best_result[
            "VALID_BIAS"
        ] = True

        return best_result

    fallback_result[
        "VALID_BIAS"
    ] = False

    return fallback_result


# ============================================================
# 15. 训练
# ============================================================
def train_epoch(
    loader,
    model,
    optimizer,
    device,
    scaler,
    use_amp,
    four_class_weights,
    subtype_weights,
    cfg,
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss_sum = 0.0
    four_loss_sum = 0.0
    binary_loss_sum = 0.0
    subtype_loss_sum = 0.0

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

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

            logits = model(
                x
            )

            (
                total_loss,
                four_loss,
                binary_loss,
                subtype_loss,
            ) = calculate_loss(
                logits,
                y,
                four_class_weights,
                subtype_weights,
                cfg,
            )

            scaled_loss = (
                total_loss
                /
                cfg[
                    "ACCUM_STEPS"
                ]
            )

        scaler.scale(
            scaled_loss
        ).backward()

        total_loss_sum += float(
            total_loss
            .detach()
            .item()
        )

        four_loss_sum += float(
            four_loss
            .detach()
            .item()
        )

        binary_loss_sum += float(
            binary_loss
            .detach()
            .item()
        )

        subtype_loss_sum += float(
            subtype_loss
            .detach()
            .item()
        )

        should_step = (
            (
                batch_index
                + 1
            )
            %
            cfg[
                "ACCUM_STEPS"
            ]
            == 0

            or

            (
                batch_index
                + 1
                ==
                len(
                    loader
                )
            )
        )

        if should_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,

                cfg[
                    "GRAD_CLIP"
                ],
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

    divisor = max(
        len(
            loader
        ),
        1,
    )

    return {
        "TOTAL": (
            total_loss_sum
            / divisor
        ),

        "FOUR": (
            four_loss_sum
            / divisor
        ),

        "BINARY": (
            binary_loss_sum
            / divisor
        ),

        "SUBTYPE": (
            subtype_loss_sum
            / divisor
        ),
    }


# ============================================================
# 16. 收集预测
# ============================================================
@torch.no_grad()
def collect_outputs(
    loader,
    model,
    device,
    four_class_weights,
    subtype_weights,
    cfg,
):

    model.eval()

    all_logits = []
    all_labels = []

    total_loss_sum = 0.0
    four_loss_sum = 0.0
    binary_loss_sum = 0.0
    subtype_loss_sum = 0.0

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        logits = model(
            x
        )

        (
            total_loss,
            four_loss,
            binary_loss,
            subtype_loss,
        ) = calculate_loss(
            logits,
            y,
            four_class_weights,
            subtype_weights,
            cfg,
        )

        all_logits.append(
            logits
            .detach()
            .cpu()
        )

        all_labels.append(
            y
            .detach()
            .cpu()
        )

        total_loss_sum += float(
            total_loss.item()
        )

        four_loss_sum += float(
            four_loss.item()
        )

        binary_loss_sum += float(
            binary_loss.item()
        )

        subtype_loss_sum += float(
            subtype_loss.item()
        )

    divisor = max(
        len(
            loader
        ),
        1,
    )

    return (
        torch.cat(
            all_logits
        ).numpy(),

        torch.cat(
            all_labels
        ).numpy(),

        {
            "TOTAL": (
                total_loss_sum
                / divisor
            ),

            "FOUR": (
                four_loss_sum
                / divisor
            ),

            "BINARY": (
                binary_loss_sum
                / divisor
            ),

            "SUBTYPE": (
                subtype_loss_sum
                / divisor
            ),
        },
    )


# ============================================================
# 17. 验证
# ============================================================
def evaluate_with_search(
    loader,
    model,
    device,
    four_class_weights,
    subtype_weights,
    cfg,
):

    (
        logits,
        labels,
        losses,
    ) = collect_outputs(
        loader,
        model,
        device,
        four_class_weights,
        subtype_weights,
        cfg,
    )

    result = search_abnormal_bias(
        logits,
        labels,
        cfg,
    )

    result[
        "LOSSES"
    ] = losses

    return result


def evaluate_fixed_bias(
    loader,
    model,
    device,
    four_class_weights,
    subtype_weights,
    cfg,
    abnormal_bias,
):

    (
        logits,
        labels,
        losses,
    ) = collect_outputs(
        loader,
        model,
        device,
        four_class_weights,
        subtype_weights,
        cfg,
    )

    prediction = np.argmax(
        apply_abnormal_bias(
            logits,
            abnormal_bias,
        ),

        axis=1,
    )

    result = calculate_metrics(
        labels,
        prediction,
    )

    result[
        "ABNORMAL_BIAS"
    ] = float(
        abnormal_bias
    )

    result[
        "LOSSES"
    ] = losses

    return result


# ============================================================
# 18. Checkpoint
# ============================================================
def serializable(
    dictionary,
):

    result = {}

    for key, value in (
        dictionary.items()
    ):

        if isinstance(
            value,
            np.ndarray,
        ):

            result[
                key
            ] = value.tolist()

        elif isinstance(
            value,
            np.bool_,
        ):

            result[
                key
            ] = bool(
                value
            )

        else:

            result[
                key
            ] = value

    return result


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    validation_result,
    cfg,
):

    torch.save(
        {
            "epoch": int(
                epoch
            ),

            "model_state": (
                model.state_dict()
            ),

            "optimizer_state": (
                optimizer.state_dict()
            ),

            "four_score": float(
                validation_result[
                    "FOUR_SCORE"
                ]
            ),

            "four_sp": float(
                validation_result[
                    "FOUR_SP"
                ]
            ),

            "four_se": float(
                validation_result[
                    "FOUR_SE"
                ]
            ),

            "macro_f1": float(
                validation_result[
                    "MACRO_F1"
                ]
            ),

            "abnormal_bias": float(
                validation_result[
                    "ABNORMAL_BIAS"
                ]
            ),

            "valid_bias": bool(
                validation_result[
                    "VALID_BIAS"
                ]
            ),

            "metrics": serializable(
                validation_result
            ),

            "config": deepcopy(
                cfg
            ),
        },

        path,
    )


# ============================================================
# 19. 主函数
# ============================================================
def main():

    cfg = CONFIG

    set_seed(
        cfg[
            "SEED"
        ]
    )

    device = torch.device(
        "cuda"
        if (
            cfg[
                "DEVICE"
            ]
            == "cuda"
            and
            torch.cuda.is_available()
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
        "[INFO] HAS_MAMBA =",
        HAS_MAMBA,
    )

    if (
        cfg[
            "REQUIRE_MAMBA"
        ]
        and
        not HAS_MAMBA
    ):

        raise RuntimeError(
            "mamba_ssm 导入失败"
        )

    root = Path(
        cfg[
            "ROOT"
        ]
    )

    train_csv = (
        root
        / "train_index.csv"
    )

    val_csv = (
        root
        / "val_index.csv"
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

    if val_csv.exists():

        selection_csv = (
            val_csv
        )

        print(
            "[INFO] 使用验证集：",
            selection_csv,
        )

    else:

        selection_csv = (
            test_csv
        )

        print(
            "[WARNING] 没有 val_index.csv，"
            "当前使用 test_index.csv "
            "选择模型和 bias。"
        )

        print(
            "[WARNING] 正式实验必须建立独立验证集。"
        )

    save_dir = Path(
        cfg[
            "SAVE_DIR"
        ]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        save_dir
        / "best.pth"
    )

    fallback_path = (
        save_dir
        / "fallback_f1.pth"
    )

    last_path = (
        save_dir
        / "last.pth"
    )

    train_set = TokenDataset(
        train_csv,
        cfg,
    )

    validation_set = TokenDataset(
        selection_csv,
        cfg,
    )

    test_set = TokenDataset(
        test_csv,
        cfg,
    )

    train_loader = make_loader(
        train_set,
        cfg,
        device,
        shuffle=True,
    )

    validation_loader = make_loader(
        validation_set,
        cfg,
        device,
        shuffle=False,
    )

    test_loader = make_loader(
        test_set,
        cfg,
        device,
        shuffle=False,
    )

    model = DirectFourClassModel(
        cfg
    ).to(
        device
    )

    print(
        "[INIT] 不加载二分类 checkpoint，"
        "不冻结 Backbone，"
        "直接训练四分类。"
    )

    with torch.no_grad():

        x, _ = next(
            iter(
                train_loader
            )
        )

        test_logits = model(
            x[
                :1
            ].to(
                device
            )
        )

        print(
            "[Shape] logits:",
            tuple(
                test_logits.shape
            ),
        )

        if tuple(
            test_logits.shape
        ) != (
            1,
            4,
        ):

            raise RuntimeError(
                "Four-class logits shape error"
            )

    four_class_weights = build_weights(
        train_set.class_counts,

        cfg[
            "FOUR_WEIGHT_POWER"
        ],

        cfg[
            "FOUR_WEIGHT_MAX"
        ],

        abnormal_only=False,
    ).to(
        device
    )

    subtype_weights = build_weights(
        train_set.class_counts,

        cfg[
            "SUBTYPE_WEIGHT_POWER"
        ],

        cfg[
            "SUBTYPE_WEIGHT_MAX"
        ],

        abnormal_only=True,
    ).to(
        device
    )

    print(
        "[Loss] four weights:",
        four_class_weights
        .detach()
        .cpu()
        .tolist(),
    )

    print(
        "[Loss] subtype weights:",
        subtype_weights
        .detach()
        .cpu()
        .tolist(),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.backbone
                    .parameters()
                ),

                "lr": (
                    cfg[
                        "BACKBONE_LR"
                    ]
                ),
            },

            {
                "params": (
                    model.head
                    .parameters()
                ),

                "lr": (
                    cfg[
                        "HEAD_LR"
                    ]
                ),
            },
        ],

        weight_decay=(
            cfg[
                "WEIGHT_DECAY"
            ]
        ),
    )

    use_amp = bool(
        cfg[
            "AMP"
        ]
        and
        device.type
        == "cuda"
    )

    scaler = make_scaler(
        use_amp
    )

    best_score = float(
        "-inf"
    )

    best_f1 = float(
        "-inf"
    )

    fallback_f1 = float(
        "-inf"
    )

    best_epoch = -1
    bad_epochs = 0
    has_valid_model = False

    print()

    print(
        "=" * 90
    )

    print(
        "DIRECT FOUR-CLASS "
        "CONSISTENT TRAINING"
    )

    print(
        "=" * 90
    )

    for epoch in range(
        1,
        cfg[
            "EPOCHS"
        ]
        + 1,
    ):

        start_time = time.time()

        current_lrs = set_lrs(
            optimizer,

            [
                cfg[
                    "BACKBONE_LR"
                ],

                cfg[
                    "HEAD_LR"
                ],
            ],

            [
                cfg[
                    "MIN_BACKBONE_LR"
                ],

                cfg[
                    "MIN_HEAD_LR"
                ],
            ],

            epoch,

            cfg[
                "EPOCHS"
            ],

            cfg[
                "WARMUP_EPOCHS"
            ],
        )

        train_result = train_epoch(
            train_loader,
            model,
            optimizer,
            device,
            scaler,
            use_amp,
            four_class_weights,
            subtype_weights,
            cfg,
        )

        validation_result = (
            evaluate_with_search(
                validation_loader,
                model,
                device,
                four_class_weights,
                subtype_weights,
                cfg,
            )
        )

        save_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            validation_result,
            cfg,
        )

        if (
            validation_result[
                "MACRO_F1"
            ]
            >
            fallback_f1
        ):

            fallback_f1 = (
                validation_result[
                    "MACRO_F1"
                ]
            )

            save_checkpoint(
                fallback_path,
                epoch,
                model,
                optimizer,
                validation_result,
                cfg,
            )

        valid_candidate = bool(
            validation_result[
                "VALID_BIAS"
            ]
        )

        improved = (
            valid_candidate
            and
            (
                validation_result[
                    "FOUR_SCORE"
                ]
                >
                best_score

                or

                (
                    validation_result[
                        "FOUR_SCORE"
                    ]
                    ==
                    best_score

                    and

                    validation_result[
                        "MACRO_F1"
                    ]
                    >
                    best_f1
                )
            )
        )

        if improved:

            best_score = (
                validation_result[
                    "FOUR_SCORE"
                ]
            )

            best_f1 = (
                validation_result[
                    "MACRO_F1"
                ]
            )

            best_epoch = (
                epoch
            )

            bad_epochs = 0
            has_valid_model = True

            marker = (
                "BEST"
            )

            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                validation_result,
                cfg,
            )

        else:

            marker = (
                "-"
            )

            if (
                epoch
                >=
                cfg[
                    "MIN_EPOCHS"
                ]
                and
                has_valid_model
            ):

                bad_epochs += 1

        print(
            f"[{marker}] "
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['EPOCHS']} | "

            f"Train "
            f"{train_result['TOTAL']:.4f} | "

            f"Four "
            f"{train_result['FOUR']:.4f} | "

            f"Bin "
            f"{train_result['BINARY']:.4f} | "

            f"Sub "
            f"{train_result['SUBTYPE']:.4f} | "

            f"Score "
            f"{validation_result['FOUR_SCORE']:.4f} | "

            f"SP "
            f"{validation_result['FOUR_SP']:.4f} | "

            f"SE "
            f"{validation_result['FOUR_SE']:.4f} | "

            f"F1 "
            f"{validation_result['MACRO_F1']:.4f} | "

            f"Bias "
            f"{validation_result['ABNORMAL_BIAS']:+.2f} | "

            f"Valid "
            f"{valid_candidate} | "

            f"Bad "
            f"{bad_epochs}/"
            f"{cfg['PATIENCE']} | "

            f"LR "
            f"{current_lrs[0]:.8f}/"
            f"{current_lrs[1]:.8f} | "

            f"{time.time() - start_time:.1f}s"
        )

        print(
            "    Recall="
            f"{np.round(validation_result['RECALL'], 3).tolist()} | "

            f"PredCount="
            f"{validation_result['PRED_COUNTS'].tolist()}"
        )

        if (
            epoch
            >=
            cfg[
                "MIN_EPOCHS"
            ]
            and
            has_valid_model
            and
            bad_epochs
            >=
            cfg[
                "PATIENCE"
            ]
        ):

            print(
                "[Early Stop] "
                f"Best epoch="
                f"{best_epoch}, "
                f"score="
                f"{best_score:.4f}"
            )

            break

    if best_path.exists():

        selected_path = (
            best_path
        )

        print(
            "[Checkpoint] 使用有效的 "
            "Score 最佳模型。"
        )

    elif fallback_path.exists():

        selected_path = (
            fallback_path
        )

        print(
            "[Checkpoint] 使用 Macro-F1 "
            "最佳备用模型。"
        )

    else:

        selected_path = (
            last_path
        )

        print(
            "[Checkpoint] 使用最后一轮模型。"
        )

    checkpoint = safe_load(
        selected_path,
        device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    test_result = evaluate_fixed_bias(
        test_loader,
        model,
        device,
        four_class_weights,
        subtype_weights,
        cfg,

        checkpoint[
            "abnormal_bias"
        ],
    )

    print()

    print(
        "=" * 80
    )

    print(
        "[FINAL]"
    )

    print(
        "=" * 80
    )

    print(
        f"Bias="
        f"{test_result['ABNORMAL_BIAS']:+.2f}"
    )

    print(
        f"Score="
        f"{test_result['FOUR_SCORE']:.4f}"
    )

    print(
        f"SP="
        f"{test_result['FOUR_SP']:.4f}"
    )

    print(
        f"SE="
        f"{test_result['FOUR_SE']:.4f}"
    )

    print(
        f"Accuracy="
        f"{test_result['ACC']:.4f}"
    )

    print(
        f"Macro-F1="
        f"{test_result['MACRO_F1']:.4f}"
    )

    print(
        "Recall=",
        np.round(
            test_result[
                "RECALL"
            ],
            4,
        ).tolist(),
    )

    print(
        "PredCount=",
        test_result[
            "PRED_COUNTS"
        ].tolist(),
    )

    print(
        "CM=\n",
        test_result[
            "FOUR_CM"
        ],
    )

    print(
        "Selected checkpoint:",
        selected_path,
    )


if __name__ == "__main__":

    main()