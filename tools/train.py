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
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. 配置
# ============================================================
CONFIG = {
    # 数据路径
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_ast_patch_tokens"
    ),

    # 保存路径
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_direct4_calibrated_v2"
    ),

    # DataLoader
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 8,
    "NUM_WORKERS": 1,

    # 训练
    "EPOCHS": 50,
    "MIN_EPOCHS": 20,
    "PATIENCE": 12,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # Backbone
    "INPUT_DIM": 768,
    "D_MODEL": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,
    "DROPOUT": 0.15,

    # 分类头
    "HEAD_DROPOUT": 0.20,

    # 学习率
    "BACKBONE_LR": 1e-5,
    "HEAD_LR": 5e-5,

    "MIN_BACKBONE_LR": 1e-6,
    "MIN_HEAD_LR": 5e-6,

    "WARMUP_EPOCHS": 3,

    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # 联合损失
    "FOUR_LOSS_WEIGHT": 1.0,

    # 降低二分类影响，避免只学 Normal / Abnormal
    "BINARY_LOSS_WEIGHT": 0.05,

    # 强化 Crackle / Wheeze / Both 区分
    "SUBTYPE_LOSS_WEIGHT": 1.0,

    "LABEL_SMOOTHING": 0.0,

    # 四分类权重
    "FOUR_WEIGHT_POWER": 0.50,
    "FOUR_WEIGHT_MAX": 2.20,

    # 异常子类权重
    "SUBTYPE_WEIGHT_POWER": 0.50,
    "SUBTYPE_WEIGHT_MAX": 2.00,

    # Normal / Abnormal 偏置搜索
    "BIAS_MIN": -1.50,
    "BIAS_MAX": 1.50,
    "BIAS_STEP": 0.05,

    # 少数类 logit 校准强度
    "TAU_MIN": 0.0,
    "TAU_MAX": 1.50,
    "TAU_STEP": 0.05,

    # 最佳模型约束
    "MIN_VALID_SP": 30.0,
    "MIN_VALID_SE": 30.0,

    # 防止 Wheeze 和 Both 再次完全塌缩
    "MIN_WHEEZE_RECALL": 0.05,
    "MIN_BOTH_RECALL": 0.02,
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
# 5. checkpoint 加载
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
# 6. Warmup + Cosine 学习率
# ============================================================
def set_epoch_lrs(
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
            base_lr * scale
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

        cosine_ratio = (
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
        input_dim=cfg[
            "INPUT_DIM"
        ],

        d_model=cfg[
            "D_MODEL"
        ],

        freq_patches=cfg[
            "FREQ_PATCHES"
        ],

        time_patches=cfg[
            "TIME_PATCHES"
        ],

        time_depth=cfg[
            "TIME_DEPTH"
        ],

        freq_depth=cfg[
            "FREQ_DEPTH"
        ],

        num_heads=cfg[
            "NHEAD"
        ],

        dropout=cfg[
            "DROPOUT"
        ],
    )


# ============================================================
# 8. 直接四分类模型
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

        missing_columns = (
            required_columns
            - set(
                self.df.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{self.csv_path} 缺少列："
                f"{sorted(missing_columns)}"
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
                "标签必须为 0、1、2、3；"
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
            f"{self.csv_path}"
        )

    def __len__(
        self,
    ):
        return len(
            self.df
        )

    def _resolve_path(
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

        token_path = self._resolve_path(
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
                f"Token shape 错误："
                f"{token_path}\n"
                f"当前="
                f"{tuple(tokens.shape)}，"
                f"要求="
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

        batch_size=cfg[
            "BATCH_SIZE"
        ],

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
# 11. 四分类类别权重
# ============================================================
def build_four_weights(
    class_counts,
    cfg,
):
    counts = np.asarray(
        class_counts,
        dtype=np.float64,
    )

    weights = np.power(
        counts[0]
        /
        np.maximum(
            counts,
            1.0,
        ),

        cfg[
            "FOUR_WEIGHT_POWER"
        ],
    )

    weights[0] = 1.0

    weights = np.clip(
        weights,
        1.0,
        cfg[
            "FOUR_WEIGHT_MAX"
        ],
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# 12. 异常子类权重
# ============================================================
def build_subtype_weights(
    class_counts,
    cfg,
):
    counts = np.asarray(
        class_counts[
            1:4
        ],
        dtype=np.float64,
    )

    weights = np.power(
        counts.max()
        /
        np.maximum(
            counts,
            1.0,
        ),

        cfg[
            "SUBTYPE_WEIGHT_POWER"
        ],
    )

    weights = np.clip(
        weights,
        1.0,
        cfg[
            "SUBTYPE_WEIGHT_MAX"
        ],
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# 13. 联合损失
#
# 所有损失都来自同一组四分类 logits，
# 不再使用多个相互冲突的分类头。
# ============================================================
def calculate_loss(
    logits,
    labels,
    four_weights,
    subtype_weights,
    cfg,
):
    # --------------------------------------------------------
    # 主任务：直接四分类
    # --------------------------------------------------------
    four_loss = F.cross_entropy(
        logits,
        labels,

        weight=four_weights,

        label_smoothing=cfg[
            "LABEL_SMOOTHING"
        ],
    )

    # --------------------------------------------------------
    # 二分类辅助任务
    #
    # Normal logit = logits[:, 0]
    # Abnormal logit = logsumexp(logits[:, 1:4])
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 异常子类辅助任务
    # --------------------------------------------------------
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

            weight=subtype_weights,

            label_smoothing=cfg[
                "LABEL_SMOOTHING"
            ],
        )

    else:

        subtype_loss = torch.zeros(
            (),
            device=logits.device,
            dtype=logits.dtype,
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

    return {
        "total": total_loss,
        "four": four_loss,
        "binary": binary_loss,
        "subtype": subtype_loss,
    }


# ============================================================
# 14. 指标
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
            cm[
                0
            ].sum()
        ),
        1,
    )

    abnormal_total = max(
        int(
            cm[
                1:
            ].sum()
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

    class_recall = recall_score(
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

        "RECALL": class_recall,

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
# 15. Logit 校准
#
# tau：
#   根据训练集类别先验，提高少数类别 logits。
#
# abnormal_bias：
#   同时调整 1、2、3 类，平衡 Normal / Abnormal。
# ============================================================
def apply_calibration(
    logits,
    abnormal_bias,
    tau,
    class_counts,
):
    adjusted_logits = logits.astype(
        np.float64,
        copy=True,
    )

    counts = np.asarray(
        class_counts,
        dtype=np.float64,
    )

    class_prior = (
        counts
        /
        counts.sum()
    )

    log_prior = np.log(
        np.clip(
            class_prior,
            1e-12,
            None,
        )
    )

    adjusted_logits = (
        adjusted_logits
        -
        float(
            tau
        )
        * log_prior.reshape(
            1,
            -1,
        )
    )

    adjusted_logits[
        :,
        1:4,
    ] += float(
        abnormal_bias
    )

    return adjusted_logits


# ============================================================
# 16. 校准结果是否合法
# ============================================================
def valid_result(
    result,
    cfg,
):
    recalls = result[
        "RECALL"
    ]

    return bool(
        result[
            "FOUR_SP"
        ]
        >= cfg[
            "MIN_VALID_SP"
        ]

        and

        result[
            "FOUR_SE"
        ]
        >= cfg[
            "MIN_VALID_SE"
        ]

        and

        recalls[
            2
        ]
        >= cfg[
            "MIN_WHEEZE_RECALL"
        ]

        and

        recalls[
            3
        ]
        >= cfg[
            "MIN_BOTH_RECALL"
        ]
    )


# ============================================================
# 17. Score 优先比较
# ============================================================
def better_score(
    current,
    best,
):
    if best is None:
        return True

    if (
        current[
            "FOUR_SCORE"
        ]
        >
        best[
            "FOUR_SCORE"
        ]
        + 1e-12
    ):
        return True

    return (
        abs(
            current[
                "FOUR_SCORE"
            ]
            -
            best[
                "FOUR_SCORE"
            ]
        )
        <= 1e-12

        and

        current[
            "MACRO_F1"
        ]
        >
        best[
            "MACRO_F1"
        ]
        + 1e-12
    )


# ============================================================
# 18. F1 优先比较
# ============================================================
def better_f1(
    current,
    best,
):
    if best is None:
        return True

    if (
        current[
            "MACRO_F1"
        ]
        >
        best[
            "MACRO_F1"
        ]
        + 1e-12
    ):
        return True

    return (
        abs(
            current[
                "MACRO_F1"
            ]
            -
            best[
                "MACRO_F1"
            ]
        )
        <= 1e-12

        and

        current[
            "FOUR_SCORE"
        ]
        >
        best[
            "FOUR_SCORE"
        ]
        + 1e-12
    )


# ============================================================
# 19. 搜索 Bias 和 Tau
# ============================================================
def search_calibration(
    logits,
    labels,
    class_counts,
    cfg,
):
    best_valid_result = None
    best_fallback_result = None

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

    tau_values = np.arange(
        cfg[
            "TAU_MIN"
        ],

        cfg[
            "TAU_MAX"
        ]
        +
        cfg[
            "TAU_STEP"
        ]
        / 2.0,

        cfg[
            "TAU_STEP"
        ],
    )

    for tau in tau_values:

        for abnormal_bias in bias_values:

            calibrated_logits = apply_calibration(
                logits,
                abnormal_bias,
                tau,
                class_counts,
            )

            prediction = np.argmax(
                calibrated_logits,
                axis=1,
            )

            current_result = calculate_metrics(
                labels,
                prediction,
            )

            current_result[
                "ABNORMAL_BIAS"
            ] = float(
                abnormal_bias
            )

            current_result[
                "TAU"
            ] = float(
                tau
            )

            # 始终保留 Macro-F1 最优的备用结果
            if better_f1(
                current_result,
                best_fallback_result,
            ):
                best_fallback_result = (
                    current_result
                )

            # 主结果需要满足所有类别约束
            if (
                valid_result(
                    current_result,
                    cfg,
                )
                and
                better_score(
                    current_result,
                    best_valid_result,
                )
            ):
                best_valid_result = (
                    current_result
                )

    if best_valid_result is not None:

        best_valid_result[
            "VALID_CALIBRATION"
        ] = True

        return best_valid_result

    if best_fallback_result is None:
        raise RuntimeError(
            "Calibration search failed."
        )

    best_fallback_result[
        "VALID_CALIBRATION"
    ] = False

    return best_fallback_result


# ============================================================
# 20. 训练一个 Epoch
# ============================================================
def train_one_epoch(
    loader,
    model,
    optimizer,
    device,
    scaler,
    use_amp,
    four_weights,
    subtype_weights,
    cfg,
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    loss_sums = {
        "total": 0.0,
        "four": 0.0,
        "binary": 0.0,
        "subtype": 0.0,
    }

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

            losses = calculate_loss(
                logits,
                y,
                four_weights,
                subtype_weights,
                cfg,
            )

            scaled_loss = (
                losses[
                    "total"
                ]
                /
                cfg[
                    "ACCUM_STEPS"
                ]
            )

        scaler.scale(
            scaled_loss
        ).backward()

        for key in loss_sums:
            loss_sums[
                key
            ] += float(
                losses[
                    key
                ]
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
            loss_sums[
                "total"
            ]
            / divisor
        ),

        "FOUR": (
            loss_sums[
                "four"
            ]
            / divisor
        ),

        "BINARY": (
            loss_sums[
                "binary"
            ]
            / divisor
        ),

        "SUBTYPE": (
            loss_sums[
                "subtype"
            ]
            / divisor
        ),
    }


# ============================================================
# 21. 收集输出
# ============================================================
@torch.no_grad()
def collect_outputs(
    loader,
    model,
    device,
    four_weights,
    subtype_weights,
    cfg,
):
    model.eval()

    all_logits = []
    all_labels = []

    loss_sums = {
        "total": 0.0,
        "four": 0.0,
        "binary": 0.0,
        "subtype": 0.0,
    }

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

        losses = calculate_loss(
            logits,
            y,
            four_weights,
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

        for key in loss_sums:
            loss_sums[
                key
            ] += float(
                losses[
                    key
                ].item()
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
                loss_sums[
                    "total"
                ]
                / divisor
            ),

            "FOUR": (
                loss_sums[
                    "four"
                ]
                / divisor
            ),

            "BINARY": (
                loss_sums[
                    "binary"
                ]
                / divisor
            ),

            "SUBTYPE": (
                loss_sums[
                    "subtype"
                ]
                / divisor
            ),
        },
    )


# ============================================================
# 22. 验证集搜索校准参数
# ============================================================
def evaluate_search(
    loader,
    model,
    device,
    four_weights,
    subtype_weights,
    class_counts,
    cfg,
):
    logits, labels, losses = collect_outputs(
        loader,
        model,
        device,
        four_weights,
        subtype_weights,
        cfg,
    )

    result = search_calibration(
        logits,
        labels,
        class_counts,
        cfg,
    )

    result[
        "LOSSES"
    ] = losses

    return result


# ============================================================
# 23. 使用固定校准参数测试
# ============================================================
def evaluate_fixed(
    loader,
    model,
    device,
    four_weights,
    subtype_weights,
    class_counts,
    abnormal_bias,
    tau,
    cfg,
):
    logits, labels, losses = collect_outputs(
        loader,
        model,
        device,
        four_weights,
        subtype_weights,
        cfg,
    )

    calibrated_logits = apply_calibration(
        logits,
        abnormal_bias,
        tau,
        class_counts,
    )

    prediction = np.argmax(
        calibrated_logits,
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
        "TAU"
    ] = float(
        tau
    )

    result[
        "LOSSES"
    ] = losses

    return result


# ============================================================
# 24. 转换为可保存类型
# ============================================================
def serializable(
    value,
):
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            key: serializable(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            serializable(
                item
            )
            for item in value
        ]

    return value


# ============================================================
# 25. 保存 checkpoint
# ============================================================
def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    result,
    four_weights,
    subtype_weights,
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
                result[
                    "FOUR_SCORE"
                ]
            ),

            "four_sp": float(
                result[
                    "FOUR_SP"
                ]
            ),

            "four_se": float(
                result[
                    "FOUR_SE"
                ]
            ),

            "macro_f1": float(
                result[
                    "MACRO_F1"
                ]
            ),

            "abnormal_bias": float(
                result[
                    "ABNORMAL_BIAS"
                ]
            ),

            "tau": float(
                result[
                    "TAU"
                ]
            ),

            "valid_calibration": bool(
                result[
                    "VALID_CALIBRATION"
                ]
            ),

            "four_weights": (
                four_weights
                .detach()
                .cpu()
                .tolist()
            ),

            "subtype_weights": (
                subtype_weights
                .detach()
                .cpu()
                .tolist()
            ),

            "metrics": serializable(
                result
            ),

            "config": deepcopy(
                cfg
            ),
        },

        path,
    )


# ============================================================
# 26. Shape Test
# ============================================================
@torch.no_grad()
def shape_test(
    loader,
    model,
    device,
):
    model.eval()

    x, _ = next(
        iter(
            loader
        )
    )

    logits = model(
        x[
            :1
        ].to(
            device
        )
    )

    print(
        "[Shape] input:",
        tuple(
            x[
                :1
            ].shape
        ),
    )

    print(
        "[Shape] logits:",
        tuple(
            logits.shape
        ),
    )

    if tuple(
        logits.shape
    ) != (
        1,
        4,
    ):
        raise RuntimeError(
            f"logits shape error："
            f"{tuple(logits.shape)}"
        )


# ============================================================
# 27. 最终结果打印
# ============================================================
def print_final(
    result,
):
    print()

    print(
        "=" * 80
    )

    print(
        "[FINAL DIRECT FOUR-CLASS TEST]"
    )

    print(
        "=" * 80
    )

    print(
        f"Bias: "
        f"{result['ABNORMAL_BIAS']:+.2f}"
    )

    print(
        f"Tau: "
        f"{result['TAU']:.2f}"
    )

    print(
        f"4-Class Score: "
        f"{result['FOUR_SCORE']:.4f}"
    )

    print(
        f"4-Class SP: "
        f"{result['FOUR_SP']:.4f}"
    )

    print(
        f"4-Class SE: "
        f"{result['FOUR_SE']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{result['ACC']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{result['MACRO_F1']:.4f}"
    )

    print(
        "Recall"
        "[Normal,Crackle,Wheeze,Both]:",

        np.round(
            result[
                "RECALL"
            ],
            4,
        ).tolist(),
    )

    print(
        "PredCount:",

        result[
            "PRED_COUNTS"
        ].tolist(),
    )

    print()

    print(
        "Four-class confusion matrix:"
    )

    print(
        result[
            "FOUR_CM"
        ]
    )

    print()

    print(
        "Binary confusion matrix:"
    )

    print(
        result[
            "BINARY_CM"
        ]
    )


# ============================================================
# 28. 主函数
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
                torch.cuda.current_device()
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
            "mamba_ssm 导入失败。"
        )

    # ========================================================
    # 数据文件
    # ========================================================
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
            "[INFO] Validation file:",
            selection_csv,
        )

    else:

        selection_csv = (
            test_csv
        )

        print(
            "[WARNING] val_index.csv 不存在，"
            "当前使用 test_index.csv "
            "选择模型和校准参数。"
        )

        print(
            "[WARNING] 正式实验必须建立独立验证集，"
            "否则存在测试集泄漏。"
        )

    # ========================================================
    # 保存目录
    # ========================================================
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
        / "best_valid_score.pth"
    )

    fallback_path = (
        save_dir
        / "best_fallback_f1.pth"
    )

    last_path = (
        save_dir
        / "last.pth"
    )

    # ========================================================
    # Dataset
    # ========================================================
    train_set = TokenDataset(
        train_csv,
        cfg,
    )

    selection_set = TokenDataset(
        selection_csv,
        cfg,
    )

    test_set = TokenDataset(
        test_csv,
        cfg,
    )

    # ========================================================
    # DataLoader
    # ========================================================
    train_loader = make_loader(
        train_set,
        cfg,
        device,
        True,
    )

    selection_loader = make_loader(
        selection_set,
        cfg,
        device,
        False,
    )

    test_loader = make_loader(
        test_set,
        cfg,
        device,
        False,
    )

    # ========================================================
    # 模型
    # ========================================================
    model = DirectFourClassModel(
        cfg
    ).to(
        device
    )

    print(
        "[INIT] 从头训练直接四分类模型；"
        "不加载二分类 checkpoint；"
        "不冻结 Backbone。"
    )

    shape_test(
        train_loader,
        model,
        device,
    )

    # ========================================================
    # 类别权重
    # ========================================================
    four_weights = (
        build_four_weights(
            train_set.class_counts,
            cfg,
        )
        .to(
            device
        )
    )

    subtype_weights = (
        build_subtype_weights(
            train_set.class_counts,
            cfg,
        )
        .to(
            device
        )
    )

    print(
        "[Loss] four weights:",

        np.round(
            four_weights
            .detach()
            .cpu()
            .numpy(),
            6,
        ).tolist(),
    )

    print(
        "[Loss] subtype weights:",

        np.round(
            subtype_weights
            .detach()
            .cpu()
            .numpy(),
            6,
        ).tolist(),
    )

    # ========================================================
    # Optimizer
    # ========================================================
    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.backbone
                    .parameters()
                ),

                "lr": cfg[
                    "BACKBONE_LR"
                ],
            },

            {
                "params": (
                    model.head
                    .parameters()
                ),

                "lr": cfg[
                    "HEAD_LR"
                ],
            },
        ],

        weight_decay=cfg[
            "WEIGHT_DECAY"
        ],
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

    # ========================================================
    # 模型选择状态
    # ========================================================
    best_score = float(
        "-inf"
    )

    best_f1 = float(
        "-inf"
    )

    best_epoch = -1

    fallback_f1 = float(
        "-inf"
    )

    fallback_score = float(
        "-inf"
    )

    bad_epochs = 0

    has_valid_model = False

    print()

    print(
        "=" * 90
    )

    print(
        "DIRECT FOUR-CLASS "
        "CALIBRATED TRAINING"
    )

    print(
        "=" * 90
    )

    # ========================================================
    # 训练循环
    # ========================================================
    for epoch in range(
        1,
        cfg[
            "EPOCHS"
        ]
        + 1,
    ):

        start_time = time.time()

        current_lrs = set_epoch_lrs(
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

        # ----------------------------------------------------
        # 训练
        # ----------------------------------------------------
        train_result = train_one_epoch(
            train_loader,
            model,
            optimizer,
            device,
            scaler,
            use_amp,
            four_weights,
            subtype_weights,
            cfg,
        )

        # ----------------------------------------------------
        # 验证并搜索校准参数
        # ----------------------------------------------------
        validation_result = evaluate_search(
            selection_loader,
            model,
            device,
            four_weights,
            subtype_weights,
            train_set.class_counts,
            cfg,
        )

        # ----------------------------------------------------
        # 永远保存最后一轮
        # ----------------------------------------------------
        save_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            validation_result,
            four_weights,
            subtype_weights,
            cfg,
        )

        current_score = validation_result[
            "FOUR_SCORE"
        ]

        current_f1 = validation_result[
            "MACRO_F1"
        ]

        # ----------------------------------------------------
        # 保存 Macro-F1 最佳备用模型
        # ----------------------------------------------------
        fallback_improved = (
            current_f1
            >
            fallback_f1
            + 1e-12

            or

            (
                abs(
                    current_f1
                    -
                    fallback_f1
                )
                <= 1e-12

                and

                current_score
                >
                fallback_score
                + 1e-12
            )
        )

        if fallback_improved:

            fallback_f1 = (
                current_f1
            )

            fallback_score = (
                current_score
            )

            save_checkpoint(
                fallback_path,
                epoch,
                model,
                optimizer,
                validation_result,
                four_weights,
                subtype_weights,
                cfg,
            )

        # ----------------------------------------------------
        # 主最佳模型
        # ----------------------------------------------------
        valid_candidate = bool(
            validation_result[
                "VALID_CALIBRATION"
            ]
        )

        main_improved = (
            valid_candidate

            and

            (
                current_score
                >
                best_score
                + 1e-12

                or

                (
                    abs(
                        current_score
                        -
                        best_score
                    )
                    <= 1e-12

                    and

                    current_f1
                    >
                    best_f1
                    + 1e-12
                )
            )
        )

        if main_improved:

            best_score = (
                current_score
            )

            best_f1 = (
                current_f1
            )

            best_epoch = (
                epoch
            )

            bad_epochs = 0

            has_valid_model = (
                True
            )

            marker = (
                "BEST"
            )

            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                validation_result,
                four_weights,
                subtype_weights,
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

        elapsed = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # 日志
        # ----------------------------------------------------
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

            f"Tau "
            f"{validation_result['TAU']:.2f} | "

            f"Valid "
            f"{valid_candidate} | "

            f"Bad "
            f"{bad_epochs}/"
            f"{cfg['PATIENCE']} | "

            f"LR "
            f"{current_lrs[0]:.8f}/"
            f"{current_lrs[1]:.8f} | "

            f"{elapsed:.1f}s"
        )

        print(
            "    Recall[0,1,2,3]="
            f"{np.round(validation_result['RECALL'], 3).tolist()} | "

            f"PredCount="
            f"{validation_result['PRED_COUNTS'].tolist()}"
        )

        # ----------------------------------------------------
        # Early Stop
        # ----------------------------------------------------
        should_early_stop = (
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
        )

        if should_early_stop:

            print(
                "[Early Stop] "
                f"Epoch="
                f"{epoch}, "
                f"Best Epoch="
                f"{best_epoch}, "
                f"Best Score="
                f"{best_score:.4f}"
            )

            break

    # ========================================================
    # 选择最终模型
    # ========================================================
    if best_path.exists():

        selected_path = (
            best_path
        )

        print(
            "[Checkpoint] 使用满足全部约束的"
            "最佳 Score 模型。"
        )

    elif fallback_path.exists():

        selected_path = (
            fallback_path
        )

        print(
            "[Checkpoint] 没有满足全部约束的模型，"
            "使用 Macro-F1 最佳模型。"
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

    # ========================================================
    # 最终测试
    # ========================================================
    final_result = evaluate_fixed(
        test_loader,
        model,
        device,
        four_weights,
        subtype_weights,
        train_set.class_counts,
        checkpoint[
            "abnormal_bias"
        ],
        checkpoint[
            "tau"
        ],
        cfg,
    )

    print(
        "[Training Completed] "
        f"Selected Epoch="
        f"{checkpoint['epoch']}, "
        f"Validation Score="
        f"{checkpoint['four_score']:.4f}, "
        f"SP="
        f"{checkpoint['four_sp']:.4f}, "
        f"SE="
        f"{checkpoint['four_se']:.4f}, "
        f"F1="
        f"{checkpoint['macro_f1']:.4f}"
    )

    print_final(
        final_result
    )

    print()

    print(
        "Selected checkpoint:",
        selected_path,
    )

    print(
        "Best valid checkpoint:",
        best_path,
    )

    print(
        "Fallback checkpoint:",
        fallback_path,
    )

    print(
        "Last checkpoint:",
        last_path,
    )


if __name__ == "__main__":

    main()