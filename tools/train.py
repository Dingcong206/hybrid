#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

    # 使用之前表现更好的二分类模型
    "BINARY_CKPT": (
        "/data/dingcong/hybrid/"
        "checkpoints_two_stage_cascade/"
        "best_binary.pth"
    ),

    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_two_stage_multilabel"
    ),

    # --------------------------------------------------------
    # 通用训练参数
    # --------------------------------------------------------
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 1,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # --------------------------------------------------------
    # Backbone 参数
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
    # Stage 2 多标签分类头
    # --------------------------------------------------------
    "HEAD_HIDDEN_DIM": 128,
    "HEAD_DROPOUT": 0.20,

    # --------------------------------------------------------
    # Stage 2 训练参数
    # --------------------------------------------------------
    "STAGE2_EPOCHS": 40,
    "STAGE2_PATIENCE": 12,

    # 前两个 Epoch 只训练分类头
    "STAGE2_HEAD_ONLY_EPOCHS": 2,

    "STAGE2_BACKBONE_LR": 2e-6,
    "STAGE2_HEAD_LR": 2e-5,

    "WEIGHT_DECAY": 1e-2,

    # pos_weight 自动根据训练集计算，
    # 为避免权重过大或过小，对其进行截断
    "POS_WEIGHT_MIN": 0.50,
    "POS_WEIGHT_MAX": 2.00,

    # --------------------------------------------------------
    # Crackle/Wheeze 阈值搜索
    # --------------------------------------------------------
    "CRACKLE_THRESHOLD_MIN": 0.20,
    "CRACKLE_THRESHOLD_MAX": 0.80,

    "WHEEZE_THRESHOLD_MIN": 0.20,
    "WHEEZE_THRESHOLD_MAX": 0.80,

    "PATHOLOGY_THRESHOLD_STEP": 0.02,
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

    torch.cuda.manual_seed(
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
# 4. AMP Scaler
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
# 5. 创建 Backbone
# ============================================================
def make_backbone(
    cfg,
):
    return TimeFrequencyEncoder(
        input_dim=(
            cfg["INPUT_DIM"]
        ),

        d_model=(
            cfg["D_MODEL"]
        ),

        freq_patches=(
            cfg["FREQ_PATCHES"]
        ),

        time_patches=(
            cfg["TIME_PATCHES"]
        ),

        time_depth=(
            cfg["TIME_DEPTH"]
        ),

        freq_depth=(
            cfg["FREQ_DEPTH"]
        ),

        num_heads=(
            cfg["NHEAD"]
        ),

        dropout=(
            cfg["DROPOUT"]
        ),
    )


# ============================================================
# 6. Dataset
# ============================================================
class TokenDataset(
    Dataset
):

    def __init__(
        self,
        csv_path,
        cfg,
        abnormal_only=False,
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
                f"{csv_path} 缺少列："
                f"{sorted(missing_columns)}；"
                f"当前列："
                f"{self.df.columns.tolist()}"
            )

        self.df["label"] = (
            self.df["label"]
            .astype(int)
        )

        # Stage 2 只训练真实异常样本
        if abnormal_only:

            self.df = (
                self.df[
                    self.df["label"] > 0
                ]
                .reset_index(
                    drop=True
                )
            )

        self.labels = (
            self.df["label"]
            .to_numpy(
                dtype=np.int64
            )
        )

        self.expected_shape = (
            cfg["FREQ_PATCHES"]
            * cfg["TIME_PATCHES"],

            cfg["INPUT_DIM"],
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

        if (
            abnormal_only
            and np.any(
                self.labels == 0
            )
        ):

            raise RuntimeError(
                "abnormal_only=True "
                "时仍包含 Normal 样本。"
            )

        self.class_counts = (
            np.bincount(
                self.labels,
                minlength=4,
            )
        )

        print(
            f"[Dataset] "
            f"samples={len(self.df)} | "
            f"abnormal_only={abnormal_only} | "
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

        token_path = (
            self._resolve_path(
                str(
                    row[
                        "tokens_path"
                    ]
                )
            )
        )

        tokens = np.load(
            token_path
        )

        if (
            tuple(
                tokens.shape
            )
            != self.expected_shape
        ):

            raise ValueError(
                f"Token shape error："
                f"{token_path}\n"
                f"当前："
                f"{tuple(tokens.shape)}；"
                f"要求："
                f"{self.expected_shape}"
            )

        x = torch.from_numpy(
            tokens
        ).float()

        y = torch.tensor(
            int(
                row["label"]
            ),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# 7. Collate
# ============================================================
def collate_fixed(
    batch,
):
    xs, ys = zip(
        *batch
    )

    return (
        torch.stack(
            xs,
            dim=0,
        ),

        torch.stack(
            ys,
            dim=0,
        ).view(-1),
    )


# ============================================================
# 8. DataLoader
# ============================================================
def make_loader(
    dataset,
    cfg,
    device,
    shuffle,
):
    workers = int(
        cfg["NUM_WORKERS"]
    )

    return DataLoader(
        dataset,

        batch_size=int(
            cfg["BATCH_SIZE"]
        ),

        shuffle=shuffle,

        num_workers=workers,

        pin_memory=(
            device.type == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        drop_last=False,

        collate_fn=collate_fixed,
    )


# ============================================================
# 9. 多标签异常分类头
#
# 输出：
#   logits[:, 0] = Crackle 是否存在
#   logits[:, 1] = Wheeze 是否存在
# ============================================================
class MultilabelAbnormalHead(
    nn.Module
):

    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                cfg["D_MODEL"]
            ),

            nn.Linear(
                cfg["D_MODEL"],
                cfg["HEAD_HIDDEN_DIM"],
            ),

            nn.GELU(),

            nn.Dropout(
                cfg["HEAD_DROPOUT"]
            ),

            nn.Linear(
                cfg["HEAD_HIDDEN_DIM"],
                2,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


# ============================================================
# 10. 标签转换
#
# 1 Crackle -> [1, 0]
# 2 Wheeze  -> [0, 1]
# 3 Both    -> [1, 1]
# ============================================================
def make_multilabel_target(
    y,
):
    target = torch.zeros(
        y.size(0),
        2,

        dtype=torch.float32,

        device=y.device,
    )

    # Crackle 或 Both
    target[:, 0] = (
        (
            y == 1
        )
        |
        (
            y == 3
        )
    ).float()

    # Wheeze 或 Both
    target[:, 1] = (
        (
            y == 2
        )
        |
        (
            y == 3
        )
    ).float()

    return target


# ============================================================
# 11. 计算 BCE pos_weight
# ============================================================
def build_pos_weight(
    labels,
    cfg,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    # Crackle 存在：
    # 正样本 = class 1 + class 3
    # 负样本 = class 2
    crackle_positive = np.sum(
        (
            labels == 1
        )
        |
        (
            labels == 3
        )
    )

    crackle_negative = np.sum(
        labels == 2
    )

    # Wheeze 存在：
    # 正样本 = class 2 + class 3
    # 负样本 = class 1
    wheeze_positive = np.sum(
        (
            labels == 2
        )
        |
        (
            labels == 3
        )
    )

    wheeze_negative = np.sum(
        labels == 1
    )

    raw_weights = np.asarray(
        [
            crackle_negative
            / max(
                crackle_positive,
                1,
            ),

            wheeze_negative
            / max(
                wheeze_positive,
                1,
            ),
        ],

        dtype=np.float32,
    )

    clipped_weights = np.clip(
        raw_weights,

        float(
            cfg[
                "POS_WEIGHT_MIN"
            ]
        ),

        float(
            cfg[
                "POS_WEIGHT_MAX"
            ]
        ),
    )

    print(
        "[Stage 2] Crackle positive/negative:",
        crackle_positive,
        crackle_negative,
    )

    print(
        "[Stage 2] Wheeze positive/negative:",
        wheeze_positive,
        wheeze_negative,
    )

    print(
        "[Stage 2] raw pos_weight:",
        raw_weights.tolist(),
    )

    print(
        "[Stage 2] clipped pos_weight:",
        clipped_weights.tolist(),
    )

    return torch.tensor(
        clipped_weights,
        dtype=torch.float32,
    )


# ============================================================
# 12. 将多标签概率映射为 1/2/3 类
# ============================================================
def pathology_probabilities_to_class(
    probabilities,
    crackle_threshold,
    wheeze_threshold,
):
    crackle_probability = (
        probabilities[:, 0]
    )

    wheeze_probability = (
        probabilities[:, 1]
    )

    has_crackle = (
        crackle_probability
        >= crackle_threshold
    )

    has_wheeze = (
        wheeze_probability
        >= wheeze_threshold
    )

    prediction = np.zeros(
        len(
            probabilities
        ),

        dtype=np.int64,
    )

    # 只有 Crackle
    prediction[
        has_crackle
        & ~has_wheeze
    ] = 1

    # 只有 Wheeze
    prediction[
        ~has_crackle
        & has_wheeze
    ] = 2

    # 同时存在
    prediction[
        has_crackle
        & has_wheeze
    ] = 3

    # 两个概率都低于阈值时，
    # 强制选择概率更高的类别，
    # 避免 Stage 1 已判为异常却输出 Normal
    neither_mask = (
        ~has_crackle
        & ~has_wheeze
    )

    prediction[
        neither_mask
        & (
            crackle_probability
            >= wheeze_probability
        )
    ] = 1

    prediction[
        neither_mask
        & (
            wheeze_probability
            > crackle_probability
        )
    ] = 2

    return prediction


# ============================================================
# 13. 二分类指标
# ============================================================
def calculate_binary_metrics(
    y_true_four,
    y_pred_binary,
):
    y_true_binary = (
        y_true_four > 0
    ).astype(
        np.int64
    )

    y_pred_binary = (
        y_pred_binary
        .astype(
            np.int64
        )
    )

    cm = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[
            0,
            1,
        ],
    )

    normal_total = float(
        cm[0].sum()
    )

    abnormal_total = float(
        cm[1].sum()
    )

    binary_sp = (
        100.0
        * float(
            cm[0, 0]
        )
        / max(
            normal_total,
            1.0,
        )
    )

    binary_se = (
        100.0
        * float(
            cm[1, 1]
        )
        / max(
            abnormal_total,
            1.0,
        )
    )

    binary_score = (
        binary_sp
        + binary_se
    ) / 2.0

    return {
        "BINARY_SP": float(
            binary_sp
        ),

        "BINARY_SE": float(
            binary_se
        ),

        "BINARY_SCORE": float(
            binary_score
        ),

        "BINARY_CM": cm,
    }


# ============================================================
# 14. 严格四分类指标
# ============================================================
def calculate_four_metrics(
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

    normal_total = float(
        cm[0].sum()
    )

    abnormal_total = float(
        cm[1:].sum()
    )

    four_sp = (
        100.0
        * float(
            cm[0, 0]
        )
        / max(
            normal_total,
            1.0,
        )
    )

    abnormal_correct = float(
        cm[1, 1]
        + cm[2, 2]
        + cm[3, 3]
    )

    four_se = (
        100.0
        * abnormal_correct
        / max(
            abnormal_total,
            1.0,
        )
    )

    four_score = (
        four_sp
        + four_se
    ) / 2.0

    accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
        * 100.0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,

        average="macro",

        zero_division=0,
    )

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

    predicted_counts = (
        np.bincount(
            y_pred,
            minlength=4,
        )
    )

    return {
        "FOUR_SP": float(
            four_sp
        ),

        "FOUR_SE": float(
            four_se
        ),

        "FOUR_SCORE": float(
            four_score
        ),

        "ACC": float(
            accuracy
        ),

        "F1": float(
            macro_f1
        ),

        "RECALL": (
            class_recall
        ),

        "PRED_COUNTS": (
            predicted_counts
        ),

        "FOUR_CM": cm,
    }


# ============================================================
# 15. 异常三分类指标
# ============================================================
def calculate_abnormal_metrics(
    y_true,
    y_pred,
):
    abnormal_mask = (
        y_true > 0
    )

    abnormal_true = y_true[
        abnormal_mask
    ]

    abnormal_pred = y_pred[
        abnormal_mask
    ]

    abnormal_accuracy = (
        accuracy_score(
            abnormal_true,
            abnormal_pred,
        )
        * 100.0
    )

    abnormal_f1 = f1_score(
        abnormal_true,
        abnormal_pred,

        labels=[
            1,
            2,
            3,
        ],

        average="macro",

        zero_division=0,
    )

    abnormal_recall = recall_score(
        abnormal_true,
        abnormal_pred,

        labels=[
            1,
            2,
            3,
        ],

        average=None,

        zero_division=0,
    )

    abnormal_cm = confusion_matrix(
        abnormal_true,
        abnormal_pred,

        labels=[
            1,
            2,
            3,
        ],
    )

    return {
        "ABNORMAL_ACC": float(
            abnormal_accuracy
        ),

        "ABNORMAL_F1": float(
            abnormal_f1
        ),

        "ABNORMAL_RECALL": (
            abnormal_recall
        ),

        "ABNORMAL_CM": (
            abnormal_cm
        ),
    }


# ============================================================
# 16. Stage 2 训练一个 Epoch
# ============================================================
def train_stage2_epoch(
    loader,
    abnormal_backbone,
    abnormal_head,
    optimizer,
    device,
    scaler,
    use_amp,
    cfg,
    freeze_backbone,
    pos_weight,
):
    if freeze_backbone:

        abnormal_backbone.eval()

    else:

        abnormal_backbone.train()

    abnormal_head.train()

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(
            device
        )
    )

    trainable_parameters = [
        parameter
        for parameter
        in (
            list(
                abnormal_backbone
                .parameters()
            )
            +
            list(
                abnormal_head
                .parameters()
            )
        )
        if parameter.requires_grad
    ]

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    optimizer_steps = 0

    number_of_batches = len(
        loader
    )

    accumulation_steps = int(
        cfg["ACCUM_STEPS"]
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

        target = make_multilabel_target(
            y
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            if freeze_backbone:

                with torch.no_grad():

                    feature = (
                        abnormal_backbone(
                            x
                        )
                    )

            else:

                feature = (
                    abnormal_backbone(
                        x
                    )
                )

            logits = abnormal_head(
                feature
            )

            raw_loss = criterion(
                logits,
                target,
            )

            loss = (
                raw_loss
                / accumulation_steps
            )

        scaler.scale(
            loss
        ).backward()

        total_loss += float(
            raw_loss
            .detach()
            .item()
        )

        should_step = (
            (
                batch_index + 1
            )
            % accumulation_steps
            == 0

            or

            (
                batch_index + 1
                == number_of_batches
            )
        )

        if should_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=2.0,
            )

            old_scale = (
                scaler.get_scale()
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            new_scale = (
                scaler.get_scale()
            )

            if new_scale >= old_scale:

                optimizer_steps += 1

            optimizer.zero_grad(
                set_to_none=True
            )

    mean_loss = (
        total_loss
        / max(
            number_of_batches,
            1,
        )
    )

    return (
        mean_loss,
        optimizer_steps,
    )


# ============================================================
# 17. 收集级联模型输出
# ============================================================
@torch.no_grad()
def collect_cascade_outputs(
    loader,
    binary_backbone,
    binary_head,
    binary_threshold,
    abnormal_backbone,
    abnormal_head,
    device,
):
    binary_backbone.eval()
    binary_head.eval()

    abnormal_backbone.eval()
    abnormal_head.eval()

    all_true = []
    all_binary_prediction = []
    all_pathology_probability = []

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------
        binary_feature = (
            binary_backbone(
                x
            )
        )

        binary_logits = (
            binary_head(
                binary_feature
            )
        )

        abnormal_probability = (
            torch.softmax(
                binary_logits,
                dim=1,
            )[:, 1]
        )

        binary_prediction = (
            abnormal_probability
            >= binary_threshold
        )

        # 为一个 Batch 中所有样本建立两个病理概率
        pathology_probability = torch.zeros(
            x.size(0),
            2,

            dtype=torch.float32,

            device=device,
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------
        if binary_prediction.any():

            abnormal_input = x[
                binary_prediction
            ]

            abnormal_feature = (
                abnormal_backbone(
                    abnormal_input
                )
            )

            abnormal_logits = (
                abnormal_head(
                    abnormal_feature
                )
            )

            pathology_probability[
                binary_prediction
            ] = torch.sigmoid(
                abnormal_logits
            )

        all_true.append(
            y.numpy()
        )

        all_binary_prediction.append(
            binary_prediction
            .long()
            .detach()
            .cpu()
            .numpy()
        )

        all_pathology_probability.append(
            pathology_probability
            .detach()
            .cpu()
            .numpy()
        )

    y_true = np.concatenate(
        all_true
    )

    y_pred_binary = np.concatenate(
        all_binary_prediction
    )

    pathology_probability = np.concatenate(
        all_pathology_probability
    )

    return (
        y_true,
        y_pred_binary,
        pathology_probability,
    )


# ============================================================
# 18. 构建最终四分类结果
# ============================================================
def build_final_prediction(
    y_pred_binary,
    pathology_probability,
    crackle_threshold,
    wheeze_threshold,
):
    final_prediction = np.zeros(
        len(
            y_pred_binary
        ),

        dtype=np.int64,
    )

    abnormal_mask = (
        y_pred_binary == 1
    )

    if np.any(
        abnormal_mask
    ):

        final_prediction[
            abnormal_mask
        ] = pathology_probabilities_to_class(
            pathology_probability[
                abnormal_mask
            ],

            crackle_threshold,

            wheeze_threshold,
        )

    return final_prediction


# ============================================================
# 19. 搜索 Crackle/Wheeze 阈值
# ============================================================
def search_pathology_thresholds(
    y_true,
    y_pred_binary,
    pathology_probability,
    cfg,
):
    crackle_thresholds = np.arange(
        float(
            cfg[
                "CRACKLE_THRESHOLD_MIN"
            ]
        ),

        float(
            cfg[
                "CRACKLE_THRESHOLD_MAX"
            ]
        )
        +
        float(
            cfg[
                "PATHOLOGY_THRESHOLD_STEP"
            ]
        )
        / 2.0,

        float(
            cfg[
                "PATHOLOGY_THRESHOLD_STEP"
            ]
        ),
    )

    wheeze_thresholds = np.arange(
        float(
            cfg[
                "WHEEZE_THRESHOLD_MIN"
            ]
        ),

        float(
            cfg[
                "WHEEZE_THRESHOLD_MAX"
            ]
        )
        +
        float(
            cfg[
                "PATHOLOGY_THRESHOLD_STEP"
            ]
        )
        / 2.0,

        float(
            cfg[
                "PATHOLOGY_THRESHOLD_STEP"
            ]
        ),
    )

    fixed_binary_metrics = (
        calculate_binary_metrics(
            y_true,
            y_pred_binary,
        )
    )

    best_metrics = None

    for crackle_threshold in crackle_thresholds:

        for wheeze_threshold in wheeze_thresholds:

            final_prediction = (
                build_final_prediction(
                    y_pred_binary,

                    pathology_probability,

                    float(
                        crackle_threshold
                    ),

                    float(
                        wheeze_threshold
                    ),
                )
            )

            current_metrics = {
                **fixed_binary_metrics,

                **calculate_four_metrics(
                    y_true,
                    final_prediction,
                ),

                **calculate_abnormal_metrics(
                    y_true,
                    final_prediction,
                ),

                "CRACKLE_THRESHOLD": float(
                    crackle_threshold
                ),

                "WHEEZE_THRESHOLD": float(
                    wheeze_threshold
                ),
            }

            if best_metrics is None:

                improved = True

            else:

                current_score = float(
                    current_metrics[
                        "FOUR_SCORE"
                    ]
                )

                best_score = float(
                    best_metrics[
                        "FOUR_SCORE"
                    ]
                )

                improved = (
                    current_score
                    > best_score
                    + 1e-12
                )

                # Four Score 相同，
                # 优先选择异常 Macro-F1 更高的阈值
                if (
                    not improved
                    and abs(
                        current_score
                        - best_score
                    )
                    <= 1e-12
                    and
                    current_metrics[
                        "ABNORMAL_F1"
                    ]
                    >
                    best_metrics[
                        "ABNORMAL_F1"
                    ]
                ):

                    improved = True

            if improved:

                best_metrics = (
                    current_metrics
                )

    if best_metrics is None:

        raise RuntimeError(
            "Crackle/Wheeze "
            "阈值搜索失败。"
        )

    return best_metrics


# ============================================================
# 20. 使用阈值搜索评价
# ============================================================
@torch.no_grad()
def evaluate_with_threshold_search(
    loader,
    binary_backbone,
    binary_head,
    binary_threshold,
    abnormal_backbone,
    abnormal_head,
    device,
    cfg,
):
    (
        y_true,
        y_pred_binary,
        pathology_probability,
    ) = collect_cascade_outputs(
        loader,

        binary_backbone,
        binary_head,

        binary_threshold,

        abnormal_backbone,
        abnormal_head,

        device,
    )

    return search_pathology_thresholds(
        y_true,

        y_pred_binary,

        pathology_probability,

        cfg,
    )


# ============================================================
# 21. 使用固定阈值评价
# ============================================================
@torch.no_grad()
def evaluate_with_fixed_thresholds(
    loader,
    binary_backbone,
    binary_head,
    binary_threshold,
    abnormal_backbone,
    abnormal_head,
    device,
    crackle_threshold,
    wheeze_threshold,
):
    (
        y_true,
        y_pred_binary,
        pathology_probability,
    ) = collect_cascade_outputs(
        loader,

        binary_backbone,
        binary_head,

        binary_threshold,

        abnormal_backbone,
        abnormal_head,

        device,
    )

    final_prediction = (
        build_final_prediction(
            y_pred_binary,

            pathology_probability,

            crackle_threshold,

            wheeze_threshold,
        )
    )

    metrics = {}

    metrics.update(
        calculate_binary_metrics(
            y_true,
            y_pred_binary,
        )
    )

    metrics.update(
        calculate_four_metrics(
            y_true,
            final_prediction,
        )
    )

    metrics.update(
        calculate_abnormal_metrics(
            y_true,
            final_prediction,
        )
    )

    metrics[
        "CRACKLE_THRESHOLD"
    ] = float(
        crackle_threshold
    )

    metrics[
        "WHEEZE_THRESHOLD"
    ] = float(
        wheeze_threshold
    )

    return metrics


# ============================================================
# 22. 指标转换
# ============================================================
def serializable_metrics(
    metrics,
):
    result = {}

    for key, value in (
        metrics.items()
    ):

        if isinstance(
            value,
            np.ndarray,
        ):

            result[key] = (
                value.tolist()
            )

        else:

            result[key] = (
                value
            )

    return result


# ============================================================
# 23. 保存模型
# ============================================================
def save_checkpoint(
    path,
    epoch,
    abnormal_backbone,
    abnormal_head,
    optimizer,
    scheduler,
    metrics,
    pos_weight,
    cfg,
):
    torch.save(
        {
            "epoch": int(
                epoch
            ),

            "abnormal_backbone_state": (
                abnormal_backbone
                .state_dict()
            ),

            "abnormal_head_state": (
                abnormal_head
                .state_dict()
            ),

            "optimizer_state": (
                optimizer.state_dict()
            ),

            "scheduler_state": (
                scheduler.state_dict()
            ),

            "four_score": float(
                metrics[
                    "FOUR_SCORE"
                ]
            ),

            "binary_score": float(
                metrics[
                    "BINARY_SCORE"
                ]
            ),

            "crackle_threshold": float(
                metrics[
                    "CRACKLE_THRESHOLD"
                ]
            ),

            "wheeze_threshold": float(
                metrics[
                    "WHEEZE_THRESHOLD"
                ]
            ),

            "pos_weight": (
                pos_weight
                .detach()
                .cpu()
                .tolist()
            ),

            "metrics": (
                serializable_metrics(
                    metrics
                )
            ),

            "config": deepcopy(
                cfg
            ),
        },

        path,
    )


# ============================================================
# 24. 打印最终结果
# ============================================================
def print_final(
    metrics,
    binary_threshold,
):
    print()

    print(
        "=" * 80
    )

    print(
        "[FINAL MULTILABEL CASCADE TEST]"
    )

    print(
        "=" * 80
    )

    print(
        f"Binary threshold: "
        f"{binary_threshold:.4f}"
    )

    print(
        f"Crackle threshold: "
        f"{metrics['CRACKLE_THRESHOLD']:.4f}"
    )

    print(
        f"Wheeze threshold: "
        f"{metrics['WHEEZE_THRESHOLD']:.4f}"
    )

    print()

    print(
        f"Binary Score: "
        f"{metrics['BINARY_SCORE']:.4f}"
    )

    print(
        f"Binary SP: "
        f"{metrics['BINARY_SP']:.4f}"
    )

    print(
        f"Binary SE: "
        f"{metrics['BINARY_SE']:.4f}"
    )

    print()

    print(
        f"4-Class Score: "
        f"{metrics['FOUR_SCORE']:.4f}"
    )

    print(
        f"4-Class SP: "
        f"{metrics['FOUR_SP']:.4f}"
    )

    print(
        f"4-Class SE: "
        f"{metrics['FOUR_SE']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{metrics['ACC']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{metrics['F1']:.4f}"
    )

    print()

    print(
        f"Abnormal Accuracy: "
        f"{metrics['ABNORMAL_ACC']:.4f}"
    )

    print(
        f"Abnormal Macro-F1: "
        f"{metrics['ABNORMAL_F1']:.4f}"
    )

    print(
        "Abnormal Recall"
        "[Crackle,Wheeze,Both]:",

        np.round(
            metrics[
                "ABNORMAL_RECALL"
            ],
            4,
        ).tolist(),
    )

    print()

    print(
        "Recall"
        "[Normal,Crackle,Wheeze,Both]:",

        np.round(
            metrics[
                "RECALL"
            ],
            4,
        ).tolist(),
    )

    print(
        "PredCount:",

        metrics[
            "PRED_COUNTS"
        ].tolist(),
    )

    print()

    print(
        "Binary confusion matrix:"
    )

    print(
        metrics[
            "BINARY_CM"
        ]
    )

    print()

    print(
        "Abnormal confusion matrix:"
    )

    print(
        metrics[
            "ABNORMAL_CM"
        ]
    )

    print()

    print(
        "Four-class confusion matrix:"
    )

    print(
        metrics[
            "FOUR_CM"
        ]
    )


# ============================================================
# 25. 主函数
# ============================================================
def main():

    cfg = CONFIG

    set_seed(
        int(
            cfg["SEED"]
        )
    )

    device = torch.device(
        "cuda"
        if (
            cfg["DEVICE"]
            == "cuda"
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
                torch.cuda.current_device()
            ),
        )

    print(
        "[INFO] HAS_MAMBA =",
        HAS_MAMBA,
    )

    if (
        cfg["REQUIRE_MAMBA"]
        and not HAS_MAMBA
    ):

        raise RuntimeError(
            "mamba_ssm 导入失败。"
        )

    # --------------------------------------------------------
    # 数据文件
    # --------------------------------------------------------
    root = Path(
        cfg["ROOT"]
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
            "暂时使用 test_index.csv "
            "选择 Epoch 和阈值。"
        )

    # --------------------------------------------------------
    # 二分类 checkpoint
    # --------------------------------------------------------
    binary_checkpoint_path = Path(
        cfg[
            "BINARY_CKPT"
        ]
    )

    if not binary_checkpoint_path.exists():

        raise FileNotFoundError(
            f"找不到二分类 checkpoint："
            f"{binary_checkpoint_path}"
        )

    # --------------------------------------------------------
    # 保存目录
    # --------------------------------------------------------
    save_directory = Path(
        cfg["SAVE_DIR"]
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        save_directory
        / "best_multilabel_abnormal.pth"
    )

    last_checkpoint_path = (
        save_directory
        / "last_multilabel_abnormal.pth"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    selection_dataset = TokenDataset(
        selection_csv,
        cfg,
        abnormal_only=False,
    )

    test_dataset = TokenDataset(
        test_csv,
        cfg,
        abnormal_only=False,
    )

    abnormal_train_dataset = TokenDataset(
        train_csv,
        cfg,
        abnormal_only=True,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    selection_loader = make_loader(
        selection_dataset,
        cfg,
        device,
        shuffle=False,
    )

    test_loader = make_loader(
        test_dataset,
        cfg,
        device,
        shuffle=False,
    )

    # 不使用 WeightedRandomSampler
    abnormal_train_loader = make_loader(
        abnormal_train_dataset,
        cfg,
        device,
        shuffle=True,
    )

    # ========================================================
    # Stage 1：加载并固定二分类模型
    # ========================================================
    binary_backbone = (
        make_backbone(
            cfg
        ).to(device)
    )

    # 必须与原 best_binary.pth 的结构一致
    binary_head = nn.Sequential(
        nn.Dropout(
            cfg["HEAD_DROPOUT"]
        ),

        nn.Linear(
            cfg["D_MODEL"],
            2,
        ),
    ).to(device)

    binary_checkpoint = torch.load(
        binary_checkpoint_path,
        map_location=device,
    )

    binary_backbone.load_state_dict(
        binary_checkpoint[
            "backbone_state"
        ]
    )

    binary_head.load_state_dict(
        binary_checkpoint[
            "binary_head_state"
        ]
    )

    binary_threshold = float(
        binary_checkpoint[
            "threshold"
        ]
    )

    binary_backbone.eval()
    binary_head.eval()

    for parameter in (
        list(
            binary_backbone
            .parameters()
        )
        +
        list(
            binary_head
            .parameters()
        )
    ):

        parameter.requires_grad = (
            False
        )

    print()

    print(
        "=" * 80
    )

    print(
        "STAGE 1: "
        "Fixed Normal / Abnormal checkpoint"
    )

    print(
        "=" * 80
    )

    print(
        f"Binary Score="
        f"{binary_checkpoint['binary_score']:.4f}, "
        f"Epoch="
        f"{binary_checkpoint['epoch']}, "
        f"Threshold="
        f"{binary_threshold:.4f}"
    )

    # ========================================================
    # Stage 2：多标签异常分类
    # ========================================================
    abnormal_backbone = (
        make_backbone(
            cfg
        ).to(device)
    )

    # 使用二分类 Backbone 初始化
    abnormal_backbone.load_state_dict(
        binary_checkpoint[
            "backbone_state"
        ]
    )

    abnormal_head = (
        MultilabelAbnormalHead(
            cfg
        ).to(device)
    )

    # 前两个 Epoch 冻结 Backbone
    for parameter in (
        abnormal_backbone.parameters()
    ):

        parameter.requires_grad = (
            False
        )

    pos_weight = build_pos_weight(
        abnormal_train_dataset.labels,
        cfg,
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    abnormal_backbone
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "STAGE2_BACKBONE_LR"
                    ]
                ),
            },

            {
                "params": (
                    abnormal_head
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "STAGE2_HEAD_LR"
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

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,

            T_max=int(
                cfg[
                    "STAGE2_EPOCHS"
                ]
            ),

            eta_min=5e-7,
        )
    )

    use_amp = bool(
        cfg["AMP"]
        and device.type == "cuda"
    )

    scaler = make_scaler(
        use_amp
    )

    best_four_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    print()

    print(
        "=" * 80
    )

    print(
        "STAGE 2: "
        "Multilabel Crackle/Wheeze presence"
    )

    print(
        "=" * 80
    )

    for epoch in range(
        1,
        int(
            cfg[
                "STAGE2_EPOCHS"
            ]
        )
        + 1,
    ):

        start_time = time.time()

        freeze_backbone = (
            epoch
            <= int(
                cfg[
                    "STAGE2_HEAD_ONLY_EPOCHS"
                ]
            )
        )

        if (
            epoch
            == int(
                cfg[
                    "STAGE2_HEAD_ONLY_EPOCHS"
                ]
            )
            + 1
        ):

            for parameter in (
                abnormal_backbone
                .parameters()
            ):

                parameter.requires_grad = (
                    True
                )

            print(
                "[Stage 2] "
                "Backbone 已解冻，"
                "开始联合微调。"
            )

        (
            train_loss,
            optimizer_steps,
        ) = train_stage2_epoch(
            abnormal_train_loader,

            abnormal_backbone,
            abnormal_head,

            optimizer,

            device,
            scaler,
            use_amp,

            cfg,

            freeze_backbone,

            pos_weight,
        )

        metrics = (
            evaluate_with_threshold_search(
                selection_loader,

                binary_backbone,
                binary_head,

                binary_threshold,

                abnormal_backbone,
                abnormal_head,

                device,

                cfg,
            )
        )

        backbone_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        head_lr = (
            optimizer
            .param_groups[1]["lr"]
        )

        if optimizer_steps > 0:

            scheduler.step()

        current_score = float(
            metrics[
                "FOUR_SCORE"
            ]
        )

        if (
            current_score
            > best_four_score
            + 1e-9
        ):

            best_four_score = (
                current_score
            )

            best_epoch = (
                epoch
            )

            bad_epochs = 0

            marker = (
                "BEST-4SCORE"
            )

            save_checkpoint(
                best_checkpoint_path,

                epoch,

                abnormal_backbone,
                abnormal_head,

                optimizer,
                scheduler,

                metrics,

                pos_weight,

                cfg,
            )

        else:

            bad_epochs += 1

            marker = "-"

        save_checkpoint(
            last_checkpoint_path,

            epoch,

            abnormal_backbone,
            abnormal_head,

            optimizer,
            scheduler,

            metrics,

            pos_weight,

            cfg,
        )

        print(
            f"[{marker}] "
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['STAGE2_EPOCHS']} | "

            f"Mode "
            f"{'Head-Only' if freeze_backbone else 'Fine-Tune'} | "

            f"train "
            f"{train_loss:.4f} | "

            f"4-Score "
            f"{metrics['FOUR_SCORE']:.4f} | "

            f"4-SP "
            f"{metrics['FOUR_SP']:.4f} | "

            f"4-SE "
            f"{metrics['FOUR_SE']:.4f} | "

            f"Abn-Acc "
            f"{metrics['ABNORMAL_ACC']:.4f} | "

            f"Abn-F1 "
            f"{metrics['ABNORMAL_F1']:.4f} | "

            f"C-Thr "
            f"{metrics['CRACKLE_THRESHOLD']:.2f} | "

            f"W-Thr "
            f"{metrics['WHEEZE_THRESHOLD']:.2f} | "

            f"B-LR "
            f"{backbone_lr:.8f} | "

            f"H-LR "
            f"{head_lr:.8f} | "

            f"{time.time() - start_time:.1f}s"
        )

        print(
            "    Recall[0,1,2,3]="
            f"{np.round(metrics['RECALL'], 3).tolist()} | "
            f"PredCount="
            f"{metrics['PRED_COUNTS'].tolist()}"
        )

        if (
            bad_epochs
            >= int(
                cfg[
                    "STAGE2_PATIENCE"
                ]
            )
        ):

            print(
                "[Early Stop] "
                f"Best 4-Class Score="
                f"{best_four_score:.4f}, "
                f"Epoch="
                f"{best_epoch}"
            )

            break

    # ========================================================
    # 加载最佳 Stage 2
    # ========================================================
    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    abnormal_backbone.load_state_dict(
        checkpoint[
            "abnormal_backbone_state"
        ]
    )

    abnormal_head.load_state_dict(
        checkpoint[
            "abnormal_head_state"
        ]
    )

    crackle_threshold = float(
        checkpoint[
            "crackle_threshold"
        ]
    )

    wheeze_threshold = float(
        checkpoint[
            "wheeze_threshold"
        ]
    )

    # ========================================================
    # 最终测试
    # ========================================================
    final_metrics = (
        evaluate_with_fixed_thresholds(
            test_loader,

            binary_backbone,
            binary_head,

            binary_threshold,

            abnormal_backbone,
            abnormal_head,

            device,

            crackle_threshold,

            wheeze_threshold,
        )
    )

    print_final(
        final_metrics,
        binary_threshold,
    )

    print()

    print(
        "Binary checkpoint:",
        binary_checkpoint_path,
    )

    print(
        "Multilabel checkpoint:",
        best_checkpoint_path,
    )


if __name__ == "__main__":
    main()