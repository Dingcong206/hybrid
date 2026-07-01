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

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
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
        "checkpoints_two_stage_cascade_v3"
    ),

    # --------------------------------------------------------
    # 通用参数
    # --------------------------------------------------------
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 1,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    "WEIGHT_DECAY": 1e-2,

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
    # MLP 分类头
    # --------------------------------------------------------
    "HEAD_HIDDEN_DIM": 128,
    "HEAD_DROPOUT": 0.20,

    # --------------------------------------------------------
    # 轻量 Token Masking
    # --------------------------------------------------------
    "TIME_MASK_MAX": 4,
    "FREQ_MASK_MAX": 1,

    # ========================================================
    # Stage 1：Normal / Abnormal
    # ========================================================
    "STAGE1_EPOCHS": 40,
    "STAGE1_PATIENCE": 12,

    # Backbone 与分类头使用不同学习率
    "STAGE1_BACKBONE_LR": 5e-6,
    "STAGE1_HEAD_LR": 1e-5,

    "STAGE1_MIN_BACKBONE_LR": 1e-6,
    "STAGE1_MIN_HEAD_LR": 2e-6,

    "STAGE1_WARMUP_EPOCHS": 3,

    # 每个 Batch 保持 Normal / Abnormal = 1:1
    "STAGE1_BALANCED_BATCH": True,

    "STAGE1_LABEL_SMOOTHING": 0.02,

    # 50% 样本使用轻量 Token Masking
    "STAGE1_AUG_PROB": 0.50,

    # --------------------------------------------------------
    # Binary threshold
    # --------------------------------------------------------
    "THRESHOLD_MIN": 0.05,
    "THRESHOLD_MAX": 0.95,

    # 更细的阈值搜索
    "THRESHOLD_STEP": 0.001,

    # ========================================================
    # Stage 2：Crackle / Wheeze / Both
    # ========================================================
    "STAGE2_EPOCHS": 40,
    "STAGE2_PATIENCE": 15,

    # 只冻结一个 Epoch
    "STAGE2_HEAD_ONLY_EPOCHS": 1,

    "STAGE2_BACKBONE_LR": 5e-6,
    "STAGE2_HEAD_LR": 2e-5,

    "STAGE2_MIN_BACKBONE_LR": 1e-6,
    "STAGE2_MIN_HEAD_LR": 2e-6,

    "STAGE2_WARMUP_EPOCHS": 2,

    # 中等强度平衡采样
    "ABNORMAL_SAMPLER_POWER": 0.5,

    "STAGE2_LABEL_SMOOTHING": 0.03,

    # Stage 2 轻量 Token Masking
    "STAGE2_AUG_PROB": 0.30,
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
# 5. Backbone
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
# 6. 两层 MLP 分类头
# ============================================================
class MLPHead(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                output_dim,
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
# 7. Warm-up + Cosine 学习率
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

        progress = (
            epoch
            / max(
                warmup_epochs,
                1,
            )
        )

        scale = (
            0.20
            + 0.80
            * progress
        )

        current_lrs = [
            base_lr
            * scale
            for base_lr
            in base_lrs
        ]

    else:

        cosine_total = max(
            total_epochs
            - warmup_epochs,
            1,
        )

        cosine_epoch = min(
            epoch
            - warmup_epochs,
            cosine_total,
        )

        cosine_ratio = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * cosine_epoch
                    / cosine_total
                )
            )
        )

        current_lrs = [
            min_lr
            + (
                base_lr
                - min_lr
            )
            * cosine_ratio
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
# 8. Token Masking
# ============================================================
def apply_patch_mask(
    x,
    cfg,
    probability,
):
    if probability <= 0.0:

        return x

    (
        batch_size,
        token_count,
        feature_dim,
    ) = x.shape

    freq_patches = int(
        cfg["FREQ_PATCHES"]
    )

    time_patches = int(
        cfg["TIME_PATCHES"]
    )

    expected_tokens = (
        freq_patches
        * time_patches
    )

    if token_count != expected_tokens:

        raise RuntimeError(
            f"Token 数量为 "
            f"{token_count}，"
            f"要求为 "
            f"{expected_tokens}。"
        )

    masked_x = (
        x.clone()
        .view(
            batch_size,
            freq_patches,
            time_patches,
            feature_dim,
        )
    )

    for sample_index in range(
        batch_size
    ):

        if (
            random.random()
            >= probability
        ):

            continue

        # ----------------------------------------------------
        # 时间轴 Mask
        # ----------------------------------------------------
        max_time_mask = min(
            int(
                cfg[
                    "TIME_MASK_MAX"
                ]
            ),
            time_patches,
        )

        if max_time_mask > 0:

            width = random.randint(
                1,
                max_time_mask,
            )

            start = random.randint(
                0,
                time_patches
                - width,
            )

            masked_x[
                sample_index,
                :,
                start:
                start + width,
                :,
            ] = 0.0

        # ----------------------------------------------------
        # 频率轴 Mask
        # ----------------------------------------------------
        max_freq_mask = min(
            int(
                cfg[
                    "FREQ_MASK_MAX"
                ]
            ),
            freq_patches,
        )

        if max_freq_mask > 0:

            width = random.randint(
                1,
                max_freq_mask,
            )

            start = random.randint(
                0,
                freq_patches
                - width,
            )

            masked_x[
                sample_index,
                start:
                start + width,
                :,
                :,
            ] = 0.0

    return masked_x.view(
        batch_size,
        token_count,
        feature_dim,
    )


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

        # Stage 2 只使用异常样本
        if abnormal_only:

            self.df = (
                self.df[
                    self.df["label"]
                    > 0
                ]
                .reset_index(
                    drop=True
                )
            )

        self.expected_shape = (
            cfg["FREQ_PATCHES"]
            * cfg["TIME_PATCHES"],

            cfg["INPUT_DIM"],
        )

        self.labels = (
            self.df["label"]
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
                "标签必须为 "
                "0、1、2、3；"
                f"发现："
                f"{invalid_labels.tolist()}"
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
            f"abnormal_only="
            f"{abnormal_only} | "
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
# 10. Collate
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
# 11. Stage 1 平衡 BatchSampler
# ============================================================
class BinaryBalancedBatchSampler(
    BatchSampler
):
    """
    BATCH_SIZE=4 时，每个 Batch 固定：

        2 个 Normal
        2 个 Abnormal
    """

    def __init__(
        self,
        labels,
        batch_size,
        seed=42,
    ):
        if (
            batch_size
            % 2
            != 0
        ):

            raise ValueError(
                "Binary balanced batch "
                "要求 batch_size 为偶数。"
            )

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        self.normal_indices = np.where(
            labels == 0
        )[0]

        self.abnormal_indices = np.where(
            labels > 0
        )[0]

        if (
            len(
                self.normal_indices
            )
            == 0
            or
            len(
                self.abnormal_indices
            )
            == 0
        ):

            raise ValueError(
                "训练集必须同时包含 "
                "Normal 和 Abnormal。"
            )

        self.batch_size = int(
            batch_size
        )

        self.half_batch = (
            self.batch_size
            // 2
        )

        self.seed = int(
            seed
        )

        self.epoch = 0

        largest_group = max(
            len(
                self.normal_indices
            ),

            len(
                self.abnormal_indices
            ),
        )

        self.number_of_batches = (
            math.ceil(
                largest_group
                / self.half_batch
            )
        )

    def __len__(
        self,
    ):
        return (
            self.number_of_batches
        )

    def __iter__(
        self,
    ):
        rng = (
            np.random
            .default_rng(
                self.seed
                + self.epoch
            )
        )

        self.epoch += 1

        normal_pool = (
            rng.permutation(
                self.normal_indices
            )
        )

        abnormal_pool = (
            rng.permutation(
                self.abnormal_indices
            )
        )

        normal_cursor = 0
        abnormal_cursor = 0

        for _ in range(
            self.number_of_batches
        ):

            if (
                normal_cursor
                + self.half_batch
                > len(
                    normal_pool
                )
            ):

                normal_pool = (
                    rng.permutation(
                        self.normal_indices
                    )
                )

                normal_cursor = 0

            if (
                abnormal_cursor
                + self.half_batch
                > len(
                    abnormal_pool
                )
            ):

                abnormal_pool = (
                    rng.permutation(
                        self.abnormal_indices
                    )
                )

                abnormal_cursor = 0

            normal_batch = normal_pool[
                normal_cursor:
                normal_cursor
                + self.half_batch
            ]

            abnormal_batch = abnormal_pool[
                abnormal_cursor:
                abnormal_cursor
                + self.half_batch
            ]

            normal_cursor += (
                self.half_batch
            )

            abnormal_cursor += (
                self.half_batch
            )

            batch = np.concatenate(
                [
                    normal_batch,
                    abnormal_batch,
                ]
            )

            rng.shuffle(
                batch
            )

            yield batch.tolist()


# ============================================================
# 12. 普通 DataLoader
# ============================================================
def make_standard_loader(
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
# 13. Stage 1 平衡 DataLoader
# ============================================================
def make_binary_train_loader(
    dataset,
    cfg,
    device,
):
    if not bool(
        cfg[
            "STAGE1_BALANCED_BATCH"
        ]
    ):

        return make_standard_loader(
            dataset,
            cfg,
            device,
            shuffle=True,
        )

    workers = int(
        cfg["NUM_WORKERS"]
    )

    batch_sampler = (
        BinaryBalancedBatchSampler(
            labels=(
                dataset.labels
            ),

            batch_size=int(
                cfg["BATCH_SIZE"]
            ),

            seed=int(
                cfg["SEED"]
            ),
        )
    )

    print(
        "[Stage 1 Loader] "
        "使用 1:1 Normal/Abnormal "
        "balanced batches。"
    )

    return DataLoader(
        dataset,

        batch_sampler=(
            batch_sampler
        ),

        num_workers=workers,

        pin_memory=(
            device.type
            == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        collate_fn=collate_fixed,
    )


# ============================================================
# 14. Stage 2 中等强度平衡采样
# ============================================================
def make_balanced_abnormal_loader(
    dataset,
    cfg,
    device,
):
    labels = dataset.labels

    if np.any(
        labels == 0
    ):

        raise RuntimeError(
            "Stage 2 sampler "
            "只能接收异常样本。"
        )

    class_counts = (
        np.bincount(
            labels,
            minlength=4,
        )[1:4]
        .astype(
            np.float64
        )
    )

    sampler_power = float(
        cfg[
            "ABNORMAL_SAMPLER_POWER"
        ]
    )

    class_weights = (
        1.0
        / np.power(
            np.maximum(
                class_counts,
                1.0,
            ),
            sampler_power,
        )
    )

    sample_weights = np.asarray(
        [
            class_weights[
                int(label)
                - 1
            ]
            for label
            in labels
        ],
        dtype=np.float64,
    )

    sampler = (
        WeightedRandomSampler(
            weights=torch.as_tensor(
                sample_weights,
                dtype=torch.double,
            ),

            num_samples=len(
                sample_weights
            ),

            replacement=True,
        )
    )

    expected_mass = (
        class_counts
        * class_weights
    )

    expected_ratio = (
        expected_mass
        / expected_mass.sum()
    )

    print(
        "[Stage 2 Sampler] "
        "class counts:",
        class_counts.tolist(),
    )

    print(
        "[Stage 2 Sampler] "
        "class weights:",
        class_weights.tolist(),
    )

    print(
        "[Stage 2 Sampler] "
        "expected ratio:",
        np.round(
            expected_ratio,
            4,
        ).tolist(),
    )

    workers = int(
        cfg["NUM_WORKERS"]
    )

    return DataLoader(
        dataset,

        batch_size=int(
            cfg["BATCH_SIZE"]
        ),

        sampler=sampler,

        shuffle=False,

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
# 15. 二分类指标
# ============================================================
def binary_metrics(
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
# 16. 四分类指标
# ============================================================
def four_class_metrics(
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

    macro_f1 = (
        f1_score(
            y_true,
            y_pred,

            average="macro",

            zero_division=0,
        )
    )

    class_recall = (
        recall_score(
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
# 17. Stage 1：训练一个 Epoch
# ============================================================
def train_binary_epoch(
    loader,
    backbone,
    binary_head,
    optimizer,
    device,
    scaler,
    use_amp,
    cfg,
):
    backbone.train()
    binary_head.train()

    criterion = (
        nn.CrossEntropyLoss(
            label_smoothing=float(
                cfg[
                    "STAGE1_LABEL_SMOOTHING"
                ]
            )
        )
    )

    parameters = (
        list(
            backbone.parameters()
        )
        + list(
            binary_head.parameters()
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    optimizer_steps = 0

    number_of_batches = len(
        loader
    )

    accum_steps = int(
        cfg["ACCUM_STEPS"]
    )

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

        x = apply_patch_mask(
            x,
            cfg,

            probability=float(
                cfg[
                    "STAGE1_AUG_PROB"
                ]
            ),
        )

        binary_target = (
            y > 0
        ).long()

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            feature = backbone(
                x
            )

            logits = binary_head(
                feature
            )

            raw_loss = criterion(
                logits,
                binary_target,
            )

            loss = (
                raw_loss
                / accum_steps
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
            % accum_steps
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
                parameters,
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
# 18. 收集二分类输出
# ============================================================
@torch.no_grad()
def collect_binary_outputs(
    loader,
    backbone,
    binary_head,
    device,
):
    backbone.eval()
    binary_head.eval()

    criterion = (
        nn.CrossEntropyLoss(
            reduction="sum"
        )
        .to(device)
    )

    all_probabilities = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        binary_target = (
            y > 0
        ).long()

        feature = backbone(
            x
        )

        logits = binary_head(
            feature
        )

        loss = criterion(
            logits,
            binary_target,
        )

        abnormal_probability = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
        )

        all_probabilities.append(
            abnormal_probability
            .detach()
            .cpu()
        )

        all_labels.append(
            y.detach().cpu()
        )

        total_loss += float(
            loss.item()
        )

        total_samples += int(
            y.size(0)
        )

    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    ).numpy()

    labels = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    mean_loss = (
        total_loss
        / max(
            total_samples,
            1,
        )
    )

    return (
        probabilities,
        labels,
        mean_loss,
    )


# ============================================================
# 19. 二分类阈值搜索
# ============================================================
def search_binary_threshold(
    probabilities,
    y_true,
    cfg,
):
    thresholds = np.arange(
        float(
            cfg[
                "THRESHOLD_MIN"
            ]
        ),

        float(
            cfg[
                "THRESHOLD_MAX"
            ]
        )
        + float(
            cfg[
                "THRESHOLD_STEP"
            ]
        )
        / 2.0,

        float(
            cfg[
                "THRESHOLD_STEP"
            ]
        ),
    )

    best_metrics = None
    best_threshold = None

    best_balance_difference = (
        float("inf")
    )

    for threshold in thresholds:

        prediction = (
            probabilities
            >= threshold
        ).astype(
            np.int64
        )

        metrics = binary_metrics(
            y_true,
            prediction,
        )

        score = float(
            metrics[
                "BINARY_SCORE"
            ]
        )

        balance_difference = abs(
            float(
                metrics[
                    "BINARY_SP"
                ]
            )
            -
            float(
                metrics[
                    "BINARY_SE"
                ]
            )
        )

        if best_metrics is None:

            improved = True

        else:

            best_score = float(
                best_metrics[
                    "BINARY_SCORE"
                ]
            )

            improved = (
                score
                > best_score
                + 1e-12
            )

            if (
                not improved
                and abs(
                    score
                    - best_score
                )
                <= 1e-12
                and
                balance_difference
                <
                best_balance_difference
            ):

                improved = True

        if improved:

            best_metrics = (
                metrics
            )

            best_threshold = (
                float(
                    threshold
                )
            )

            best_balance_difference = (
                balance_difference
            )

    if best_metrics is None:

        raise RuntimeError(
            "二分类阈值搜索失败。"
        )

    best_metrics[
        "THRESHOLD"
    ] = best_threshold

    return best_metrics


# ============================================================
# 20. Stage 1 验证
# ============================================================
@torch.no_grad()
def evaluate_binary(
    loader,
    backbone,
    binary_head,
    device,
    cfg,
):
    (
        probabilities,
        labels,
        loss,
    ) = collect_binary_outputs(
        loader,
        backbone,
        binary_head,
        device,
    )

    metrics = (
        search_binary_threshold(
            probabilities,
            labels,
            cfg,
        )
    )

    metrics["LOSS"] = float(
        loss
    )

    return metrics


# ============================================================
# 21. Stage 2 训练一个 Epoch
# ============================================================
def train_abnormal_epoch(
    loader,
    abnormal_backbone,
    abnormal_head,
    optimizer,
    device,
    scaler,
    use_amp,
    cfg,
    freeze_backbone,
):
    if freeze_backbone:

        abnormal_backbone.eval()

    else:

        abnormal_backbone.train()

    abnormal_head.train()

    criterion = (
        nn.CrossEntropyLoss(
            label_smoothing=float(
                cfg[
                    "STAGE2_LABEL_SMOOTHING"
                ]
            )
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

    accum_steps = int(
        cfg["ACCUM_STEPS"]
    )

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

        x = apply_patch_mask(
            x,
            cfg,

            probability=float(
                cfg[
                    "STAGE2_AUG_PROB"
                ]
            ),
        )

        abnormal_target = (
            y - 1
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
                abnormal_target,
            )

            loss = (
                raw_loss
                / accum_steps
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
            % accum_steps
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
# 22. 单独评价异常三分类
# ============================================================
@torch.no_grad()
def evaluate_abnormal(
    loader,
    abnormal_backbone,
    abnormal_head,
    device,
):
    abnormal_backbone.eval()
    abnormal_head.eval()

    all_true = []
    all_pred = []

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        abnormal_target = (
            y.numpy()
            - 1
        )

        feature = (
            abnormal_backbone(
                x
            )
        )

        logits = abnormal_head(
            feature
        )

        prediction = (
            torch.argmax(
                logits,
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        all_true.append(
            abnormal_target
        )

        all_pred.append(
            prediction
        )

    y_true = np.concatenate(
        all_true
    )

    y_pred = np.concatenate(
        all_pred
    )

    abnormal_accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
        * 100.0
    )

    abnormal_f1 = (
        f1_score(
            y_true,
            y_pred,

            average="macro",

            zero_division=0,
        )
    )

    abnormal_recall = (
        recall_score(
            y_true,
            y_pred,

            labels=[
                0,
                1,
                2,
            ],

            average=None,

            zero_division=0,
        )
    )

    abnormal_cm = (
        confusion_matrix(
            y_true,
            y_pred,

            labels=[
                0,
                1,
                2,
            ],
        )
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
# 23. 完整级联评价
# ============================================================
@torch.no_grad()
def evaluate_cascade(
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
    all_binary_pred = []
    all_four_pred = []

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

        final_prediction = (
            torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=device,
            )
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

            abnormal_prediction = (
                torch.argmax(
                    abnormal_logits,
                    dim=1,
                )
                + 1
            )

            final_prediction[
                binary_prediction
            ] = abnormal_prediction

        all_true.append(
            y.numpy()
        )

        all_binary_pred.append(
            binary_prediction
            .long()
            .detach()
            .cpu()
            .numpy()
        )

        all_four_pred.append(
            final_prediction
            .detach()
            .cpu()
            .numpy()
        )

    y_true = np.concatenate(
        all_true
    )

    y_pred_binary = (
        np.concatenate(
            all_binary_pred
        )
    )

    y_pred_four = (
        np.concatenate(
            all_four_pred
        )
    )

    metrics = {}

    metrics.update(
        binary_metrics(
            y_true,
            y_pred_binary,
        )
    )

    metrics.update(
        four_class_metrics(
            y_true,
            y_pred_four,
        )
    )

    metrics[
        "THRESHOLD"
    ] = float(
        binary_threshold
    )

    return metrics


# ============================================================
# 24. 转换指标
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
# 25. 保存 Stage 1
# ============================================================
def save_stage1(
    path,
    epoch,
    backbone,
    binary_head,
    optimizer,
    metrics,
    cfg,
):
    torch.save(
        {
            "epoch": int(
                epoch
            ),

            "backbone_state": (
                backbone.state_dict()
            ),

            "binary_head_state": (
                binary_head.state_dict()
            ),

            "optimizer_state": (
                optimizer.state_dict()
            ),

            "binary_score": float(
                metrics[
                    "BINARY_SCORE"
                ]
            ),

            "threshold": float(
                metrics[
                    "THRESHOLD"
                ]
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
# 26. 保存 Stage 2
# ============================================================
def save_stage2(
    path,
    epoch,
    abnormal_backbone,
    abnormal_head,
    optimizer,
    cascade_metrics,
    abnormal_metrics,
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

            "four_score": float(
                cascade_metrics[
                    "FOUR_SCORE"
                ]
            ),

            "binary_score": float(
                cascade_metrics[
                    "BINARY_SCORE"
                ]
            ),

            "cascade_metrics": (
                serializable_metrics(
                    cascade_metrics
                )
            ),

            "abnormal_metrics": (
                serializable_metrics(
                    abnormal_metrics
                )
            ),

            "config": deepcopy(
                cfg
            ),
        },
        path,
    )


# ============================================================
# 27. 最终结果
# ============================================================
def print_final(
    metrics,
):
    print()

    print(
        "=" * 80
    )

    print(
        "[FINAL CASCADE TEST]"
    )

    print(
        "=" * 80
    )

    print(
        f"Binary threshold: "
        f"{metrics['THRESHOLD']:.4f}"
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

    print()

    print(
        f"Accuracy: "
        f"{metrics['ACC']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{metrics['F1']:.4f}"
    )

    print(
        "Recall[0,1,2,3]:",
        np.round(
            metrics["RECALL"],
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
        "Four-class confusion matrix:"
    )

    print(
        metrics[
            "FOUR_CM"
        ]
    )


# ============================================================
# 28. 主函数
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
        cfg["REQUIRE_MAMBA"]
        and
        not HAS_MAMBA
    ):

        raise RuntimeError(
            "mamba_ssm 导入失败。"
        )

    # --------------------------------------------------------
    # 路径
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
            "选择模型和阈值。"
        )

    save_dir = Path(
        cfg["SAVE_DIR"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_binary_path = (
        save_dir
        / "best_binary.pth"
    )

    last_binary_path = (
        save_dir
        / "last_binary.pth"
    )

    best_abnormal_path = (
        save_dir
        / "best_abnormal.pth"
    )

    last_abnormal_path = (
        save_dir
        / "last_abnormal.pth"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_set = TokenDataset(
        train_csv,
        cfg,
        abnormal_only=False,
    )

    selection_set = TokenDataset(
        selection_csv,
        cfg,
        abnormal_only=False,
    )

    test_set = TokenDataset(
        test_csv,
        cfg,
        abnormal_only=False,
    )

    abnormal_train_set = (
        TokenDataset(
            train_csv,
            cfg,
            abnormal_only=True,
        )
    )

    abnormal_selection_set = (
        TokenDataset(
            selection_csv,
            cfg,
            abnormal_only=True,
        )
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_loader = (
        make_binary_train_loader(
            train_set,
            cfg,
            device,
        )
    )

    selection_loader = (
        make_standard_loader(
            selection_set,
            cfg,
            device,
            shuffle=False,
        )
    )

    test_loader = (
        make_standard_loader(
            test_set,
            cfg,
            device,
            shuffle=False,
        )
    )

    abnormal_train_loader = (
        make_balanced_abnormal_loader(
            abnormal_train_set,
            cfg,
            device,
        )
    )

    abnormal_selection_loader = (
        make_standard_loader(
            abnormal_selection_set,
            cfg,
            device,
            shuffle=False,
        )
    )

    use_amp = bool(
        cfg["AMP"]
        and
        device.type == "cuda"
    )

    # ========================================================
    # Stage 1
    # ========================================================
    print()

    print(
        "=" * 90
    )

    print(
        "STAGE 1: "
        "Normal / Abnormal"
    )

    print(
        "=" * 90
    )

    binary_backbone = (
        make_backbone(
            cfg
        ).to(device)
    )

    binary_head = MLPHead(
        input_dim=int(
            cfg["D_MODEL"]
        ),

        hidden_dim=int(
            cfg[
                "HEAD_HIDDEN_DIM"
            ]
        ),

        output_dim=2,

        dropout=float(
            cfg[
                "HEAD_DROPOUT"
            ]
        ),
    ).to(device)

    binary_optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params": (
                        binary_backbone
                        .parameters()
                    ),

                    "lr": float(
                        cfg[
                            "STAGE1_BACKBONE_LR"
                        ]
                    ),
                },

                {
                    "params": (
                        binary_head
                        .parameters()
                    ),

                    "lr": float(
                        cfg[
                            "STAGE1_HEAD_LR"
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
    )

    binary_scaler = (
        make_scaler(
            use_amp
        )
    )

    best_binary_score = -1.0
    best_binary_epoch = -1

    bad_epochs = 0

    for epoch in range(
        1,
        int(
            cfg[
                "STAGE1_EPOCHS"
            ]
        )
        + 1,
    ):

        start_time = time.time()

        current_lrs = (
            set_epoch_lrs(
                optimizer=(
                    binary_optimizer
                ),

                base_lrs=[
                    float(
                        cfg[
                            "STAGE1_BACKBONE_LR"
                        ]
                    ),

                    float(
                        cfg[
                            "STAGE1_HEAD_LR"
                        ]
                    ),
                ],

                min_lrs=[
                    float(
                        cfg[
                            "STAGE1_MIN_BACKBONE_LR"
                        ]
                    ),

                    float(
                        cfg[
                            "STAGE1_MIN_HEAD_LR"
                        ]
                    ),
                ],

                epoch=epoch,

                total_epochs=int(
                    cfg[
                        "STAGE1_EPOCHS"
                    ]
                ),

                warmup_epochs=int(
                    cfg[
                        "STAGE1_WARMUP_EPOCHS"
                    ]
                ),
            )
        )

        (
            train_loss,
            _,
        ) = train_binary_epoch(
            train_loader,

            binary_backbone,
            binary_head,

            binary_optimizer,

            device,
            binary_scaler,
            use_amp,

            cfg,
        )

        metrics = evaluate_binary(
            selection_loader,

            binary_backbone,
            binary_head,

            device,
            cfg,
        )

        if (
            metrics[
                "BINARY_SCORE"
            ]
            > best_binary_score
            + 1e-9
        ):

            best_binary_score = float(
                metrics[
                    "BINARY_SCORE"
                ]
            )

            best_binary_epoch = (
                epoch
            )

            bad_epochs = 0

            marker = (
                "BEST-BINARY"
            )

            save_stage1(
                best_binary_path,

                epoch,

                binary_backbone,
                binary_head,

                binary_optimizer,

                metrics,

                cfg,
            )

        else:

            bad_epochs += 1

            marker = "-"

        save_stage1(
            last_binary_path,

            epoch,

            binary_backbone,
            binary_head,

            binary_optimizer,

            metrics,

            cfg,
        )

        print(
            f"[{marker}] "
            f"S1 Epoch "
            f"{epoch:03d}/"
            f"{cfg['STAGE1_EPOCHS']} | "

            f"train "
            f"{train_loss:.4f} | "

            f"val "
            f"{metrics['LOSS']:.4f} | "

            f"Binary-Score "
            f"{metrics['BINARY_SCORE']:.4f} | "

            f"SP "
            f"{metrics['BINARY_SP']:.4f} | "

            f"SE "
            f"{metrics['BINARY_SE']:.4f} | "

            f"Thr "
            f"{metrics['THRESHOLD']:.3f} | "

            f"B-LR "
            f"{current_lrs[0]:.8f} | "

            f"H-LR "
            f"{current_lrs[1]:.8f} | "

            f"{time.time() - start_time:.1f}s"
        )

        if (
            bad_epochs
            >= int(
                cfg[
                    "STAGE1_PATIENCE"
                ]
            )
        ):

            print(
                "[Stage 1 Early Stop] "
                f"Best Binary Score="
                f"{best_binary_score:.4f}, "
                f"Epoch="
                f"{best_binary_epoch}"
            )

            break

    # --------------------------------------------------------
    # 加载最佳 Stage 1
    # --------------------------------------------------------
    binary_checkpoint = torch.load(
        best_binary_path,
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

    print(
        f"[Stage 1 完成] "
        f"Binary Score="
        f"{binary_checkpoint['binary_score']:.4f}, "
        f"Epoch="
        f"{binary_checkpoint['epoch']}, "
        f"Threshold="
        f"{binary_threshold:.4f}"
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

    # ========================================================
    # Stage 2
    # ========================================================
    print()

    print(
        "=" * 90
    )

    print(
        "STAGE 2: "
        "Crackle / Wheeze / Both"
    )

    print(
        "=" * 90
    )

    abnormal_backbone = (
        make_backbone(
            cfg
        ).to(device)
    )

    # 使用 Stage 1 最佳 Backbone 初始化
    abnormal_backbone.load_state_dict(
        binary_checkpoint[
            "backbone_state"
        ]
    )

    abnormal_head = MLPHead(
        input_dim=int(
            cfg["D_MODEL"]
        ),

        hidden_dim=int(
            cfg[
                "HEAD_HIDDEN_DIM"
            ]
        ),

        output_dim=3,

        dropout=float(
            cfg[
                "HEAD_DROPOUT"
            ]
        ),
    ).to(device)

    # 第 1 个 Epoch 冻结 Backbone
    for parameter in (
        abnormal_backbone
        .parameters()
    ):

        parameter.requires_grad = (
            False
        )

    abnormal_optimizer = (
        torch.optim.AdamW(
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
    )

    abnormal_scaler = (
        make_scaler(
            use_amp
        )
    )

    best_four_score = -1.0
    best_stage2_epoch = -1

    bad_epochs = 0

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

        # 第 2 个 Epoch 解冻
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

        current_lrs = (
            set_epoch_lrs(
                optimizer=(
                    abnormal_optimizer
                ),

                base_lrs=[
                    float(
                        cfg[
                            "STAGE2_BACKBONE_LR"
                        ]
                    ),

                    float(
                        cfg[
                            "STAGE2_HEAD_LR"
                        ]
                    ),
                ],

                min_lrs=[
                    float(
                        cfg[
                            "STAGE2_MIN_BACKBONE_LR"
                        ]
                    ),

                    float(
                        cfg[
                            "STAGE2_MIN_HEAD_LR"
                        ]
                    ),
                ],

                epoch=epoch,

                total_epochs=int(
                    cfg[
                        "STAGE2_EPOCHS"
                    ]
                ),

                warmup_epochs=int(
                    cfg[
                        "STAGE2_WARMUP_EPOCHS"
                    ]
                ),
            )
        )

        (
            train_loss,
            _,
        ) = train_abnormal_epoch(
            abnormal_train_loader,

            abnormal_backbone,
            abnormal_head,

            abnormal_optimizer,

            device,
            abnormal_scaler,
            use_amp,

            cfg,

            freeze_backbone,
        )

        abnormal_metrics = (
            evaluate_abnormal(
                abnormal_selection_loader,

                abnormal_backbone,
                abnormal_head,

                device,
            )
        )

        cascade_metrics = (
            evaluate_cascade(
                selection_loader,

                binary_backbone,
                binary_head,

                binary_threshold,

                abnormal_backbone,
                abnormal_head,

                device,
            )
        )

        if (
            cascade_metrics[
                "FOUR_SCORE"
            ]
            > best_four_score
            + 1e-9
        ):

            best_four_score = float(
                cascade_metrics[
                    "FOUR_SCORE"
                ]
            )

            best_stage2_epoch = (
                epoch
            )

            bad_epochs = 0

            marker = (
                "BEST-4SCORE"
            )

            save_stage2(
                best_abnormal_path,

                epoch,

                abnormal_backbone,
                abnormal_head,

                abnormal_optimizer,

                cascade_metrics,
                abnormal_metrics,

                cfg,
            )

        else:

            bad_epochs += 1

            marker = "-"

        save_stage2(
            last_abnormal_path,

            epoch,

            abnormal_backbone,
            abnormal_head,

            abnormal_optimizer,

            cascade_metrics,
            abnormal_metrics,

            cfg,
        )

        print(
            f"[{marker}] "
            f"S2 Epoch "
            f"{epoch:03d}/"
            f"{cfg['STAGE2_EPOCHS']} | "

            f"Mode "
            f"{'Head-Only' if freeze_backbone else 'Fine-Tune'} | "

            f"train "
            f"{train_loss:.4f} | "

            f"Abn-Acc "
            f"{abnormal_metrics['ABNORMAL_ACC']:.4f} | "

            f"Abn-F1 "
            f"{abnormal_metrics['ABNORMAL_F1']:.4f} | "

            f"Abn-Recall "
            f"{np.round(abnormal_metrics['ABNORMAL_RECALL'], 3).tolist()} | "

            f"4-Score "
            f"{cascade_metrics['FOUR_SCORE']:.4f} | "

            f"4-SP "
            f"{cascade_metrics['FOUR_SP']:.4f} | "

            f"4-SE "
            f"{cascade_metrics['FOUR_SE']:.4f} | "

            f"Binary-Score "
            f"{cascade_metrics['BINARY_SCORE']:.4f} | "

            f"B-LR "
            f"{current_lrs[0]:.8f} | "

            f"H-LR "
            f"{current_lrs[1]:.8f} | "

            f"{time.time() - start_time:.1f}s"
        )

        print(
            "    Recall[0,1,2,3]="
            f"{np.round(cascade_metrics['RECALL'], 3).tolist()} | "
            f"PredCount="
            f"{cascade_metrics['PRED_COUNTS'].tolist()}"
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
                "[Stage 2 Early Stop] "
                f"Best 4-Class Score="
                f"{best_four_score:.4f}, "
                f"Epoch="
                f"{best_stage2_epoch}"
            )

            break

    # --------------------------------------------------------
    # 加载最佳 Stage 2
    # --------------------------------------------------------
    abnormal_checkpoint = torch.load(
        best_abnormal_path,
        map_location=device,
    )

    abnormal_backbone.load_state_dict(
        abnormal_checkpoint[
            "abnormal_backbone_state"
        ]
    )

    abnormal_head.load_state_dict(
        abnormal_checkpoint[
            "abnormal_head_state"
        ]
    )

    print(
        f"[Stage 2 完成] "
        f"4-Class Score="
        f"{abnormal_checkpoint['four_score']:.4f}, "
        f"Epoch="
        f"{abnormal_checkpoint['epoch']}"
    )

    # --------------------------------------------------------
    # 最终测试
    # --------------------------------------------------------
    final_metrics = (
        evaluate_cascade(
            test_loader,

            binary_backbone,
            binary_head,

            binary_threshold,

            abnormal_backbone,
            abnormal_head,

            device,
        )
    )

    print_final(
        final_metrics
    )

    print()

    print(
        "Best binary checkpoint:",
        best_binary_path,
    )

    print(
        "Best abnormal checkpoint:",
        best_abnormal_path,
    )


if __name__ == "__main__":
    main()