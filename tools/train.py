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
    WeightedRandomSampler,
)


# ============================================================
# 配置
# ============================================================
CONFIG = {
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_ast_patch_tokens"
    ),

    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_two_stage_cascade_v2"
    ),

    # 已经有上一轮 best_binary.pth 时，直接复用 Stage 1。
    # 若需要重新训练二分类模型，改成 False。
    "REUSE_STAGE1_IF_EXISTS": True,

    "PREVIOUS_BINARY_CKPT": (
        "/data/dingcong/hybrid/"
        "checkpoints_two_stage_cascade/"
        "best_binary.pth"
    ),

    # 通用参数
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 1,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    "WEIGHT_DECAY": 1e-2,

    # Backbone
    "INPUT_DIM": 768,
    "D_MODEL": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,
    "DROPOUT": 0.15,
    "HEAD_DROPOUT": 0.20,

    # ========================================================
    # Stage 1：Normal / Abnormal
    # ========================================================
    "STAGE1_EPOCHS": 30,
    "STAGE1_LR": 1e-5,
    "STAGE1_PATIENCE": 10,

    # 二分类阈值只根据 Binary Score 搜索
    "THRESHOLD_MIN": 0.05,
    "THRESHOLD_MAX": 0.95,
    "THRESHOLD_STEP": 0.01,

    # ========================================================
    # Stage 2：Crackle / Wheeze / Both
    # ========================================================
    "STAGE2_EPOCHS": 40,

    # 前 3 个 Epoch 只训练异常分类头
    "STAGE2_HEAD_ONLY_EPOCHS": 3,

    # 第 4 个 Epoch 开始解冻 Backbone
    "STAGE2_BACKBONE_LR": 2e-6,

    # 异常分类头使用更高学习率
    "STAGE2_HEAD_LR": 5e-5,

    "STAGE2_PATIENCE": 15,

    # 1.0 表示完全按照异常类别频数倒数进行平衡采样
    "ABNORMAL_SAMPLER_POWER": 1.0,
}


# ============================================================
# 导入模型
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
# 随机种子
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
# AMP Scaler
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
# 创建 Backbone
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
# Dataset
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

        # Stage 2 只使用真实异常样本
        if abnormal_only:

            self.df = (
                self.df[
                    self.df["label"] > 0
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
                (self.labels < 0)
                | (self.labels > 3)
            ]
        )

        if len(
            invalid_labels
        ) > 0:

            raise ValueError(
                "标签必须为 0、1、2、3；"
                f"发现无效标签："
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
# Collate
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
# 普通 DataLoader
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
# Stage 2 平衡采样 DataLoader
# ============================================================
def make_balanced_abnormal_loader(
    dataset,
    cfg,
    device,
):
    labels = (
        dataset.labels
    )

    if np.any(
        labels == 0
    ):

        raise RuntimeError(
            "Stage 2 平衡采样器"
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

    power = float(
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
            power,
        )
    )

    sample_weights = np.asarray(
        [
            class_weights[
                int(label) - 1
            ]
            for label in labels
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

    workers = int(
        cfg["NUM_WORKERS"]
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

    return DataLoader(
        dataset,

        batch_size=int(
            cfg["BATCH_SIZE"]
        ),

        sampler=sampler,

        shuffle=False,

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
# 二分类指标
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
# 严格四分类指标
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

        "RECALL": class_recall,

        "PRED_COUNTS": (
            predicted_counts
        ),

        "FOUR_CM": cm,
    }


# ============================================================
# Stage 1：训练一个 Epoch
# ============================================================
def train_binary_epoch(
    loader,
    backbone,
    binary_head,
    optimizer,
    device,
    scaler,
    use_amp,
    accum_steps,
):
    backbone.train()
    binary_head.train()

    criterion = (
        nn.CrossEntropyLoss()
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

        # 0 = Normal
        # 1 = Abnormal
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
                max_norm=5.0,
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
# 收集二分类概率
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
# Stage 1：搜索二分类最佳阈值
# ============================================================
def search_binary_threshold(
    probabilities,
    y_true,
    cfg,
):
    thresholds = np.arange(
        cfg[
            "THRESHOLD_MIN"
        ],

        cfg[
            "THRESHOLD_MAX"
        ]
        + cfg[
            "THRESHOLD_STEP"
        ]
        / 2.0,

        cfg[
            "THRESHOLD_STEP"
        ],
    )

    best_metrics = None
    best_threshold = None

    best_balance = float(
        "inf"
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

        score = (
            metrics[
                "BINARY_SCORE"
            ]
        )

        balance = abs(
            metrics["BINARY_SP"]
            - metrics["BINARY_SE"]
        )

        if best_metrics is None:

            improved = True

        else:

            best_score = (
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
                ) <= 1e-12
                and balance
                < best_balance
            ):

                improved = True

        if improved:

            best_metrics = metrics

            best_threshold = float(
                threshold
            )

            best_balance = (
                balance
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
# Stage 1：验证
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
# Stage 2：训练一个 Epoch
# ============================================================
def train_abnormal_epoch(
    loader,
    abnormal_backbone,
    abnormal_head,
    optimizer,
    device,
    scaler,
    use_amp,
    accum_steps,
    freeze_backbone,
):
    if freeze_backbone:

        abnormal_backbone.eval()

    else:

        abnormal_backbone.train()

    abnormal_head.train()

    # 已经使用平衡采样，
    # 此处不要再叠加类别权重
    criterion = (
        nn.CrossEntropyLoss()
    )

    trainable_parameters = [
        parameter
        for parameter in (
            list(
                abnormal_backbone
                .parameters()
            )
            + list(
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

        # 1/2/3 转成 0/1/2
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
                max_norm=5.0,
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
# 单独评价异常三分类模型
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
            y.numpy() - 1
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
# 完整级联评价
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

        # ====================================================
        # Stage 1：Normal / Abnormal
        # ====================================================
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

        # 默认预测 Normal
        final_prediction = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=device,
        )

        # ====================================================
        # Stage 2：只处理第一阶段预测为异常的样本
        # ====================================================
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
# 保存 Stage 1
# ============================================================
def save_stage1(
    path,
    epoch,
    backbone,
    binary_head,
    optimizer,
    scheduler,
    metrics,
    cfg,
):
    torch.save(
        {
            "epoch": epoch,

            "backbone_state": (
                backbone.state_dict()
            ),

            "binary_head_state": (
                binary_head.state_dict()
            ),

            "optimizer_state": (
                optimizer.state_dict()
            ),

            "scheduler_state": (
                scheduler.state_dict()
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

            "config": deepcopy(
                cfg
            ),
        },
        path,
    )


# ============================================================
# 保存 Stage 2
# ============================================================
def save_stage2(
    path,
    epoch,
    abnormal_backbone,
    abnormal_head,
    optimizer,
    scheduler,
    cascade_metrics,
    abnormal_metrics,
    cfg,
):
    torch.save(
        {
            "epoch": epoch,

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
                cascade_metrics[
                    "FOUR_SCORE"
                ]
            ),

            "binary_score": float(
                cascade_metrics[
                    "BINARY_SCORE"
                ]
            ),

            "abnormal_acc": float(
                abnormal_metrics[
                    "ABNORMAL_ACC"
                ]
            ),

            "abnormal_f1": float(
                abnormal_metrics[
                    "ABNORMAL_F1"
                ]
            ),

            "config": deepcopy(
                cfg
            ),
        },
        path,
    )


# ============================================================
# 输出最终结果
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
# 主函数
# ============================================================
def main():

    cfg = CONFIG

    set_seed(
        cfg["SEED"]
    )

    device = torch.device(
        "cuda"
        if (
            cfg["DEVICE"] == "cuda"
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

    # ========================================================
    # 数据路径
    # ========================================================
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
            "[WARNING] 没有 "
            "val_index.csv，"
            "暂时使用 test_index.csv "
            "选模型和阈值。"
        )

    # ========================================================
    # 保存目录
    # ========================================================
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

    # ========================================================
    # Dataset
    # ========================================================
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

    # ========================================================
    # DataLoader
    # ========================================================
    train_loader = make_loader(
        train_set,
        cfg,
        device,
        shuffle=True,
    )

    selection_loader = (
        make_loader(
            selection_set,
            cfg,
            device,
            shuffle=False,
        )
    )

    test_loader = make_loader(
        test_set,
        cfg,
        device,
        shuffle=False,
    )

    # Stage 2 使用平衡采样
    abnormal_train_loader = (
        make_balanced_abnormal_loader(
            abnormal_train_set,
            cfg,
            device,
        )
    )

    # 验证集不能平衡采样
    abnormal_selection_loader = (
        make_loader(
            abnormal_selection_set,
            cfg,
            device,
            shuffle=False,
        )
    )

    use_amp = bool(
        cfg["AMP"]
        and device.type == "cuda"
    )

    # ========================================================
    # Stage 1
    # ========================================================
    print()
    print(
        "=" * 80
    )

    print(
        "STAGE 1: "
        "Normal / Abnormal"
    )

    print(
        "=" * 80
    )

    binary_backbone = (
        make_backbone(
            cfg
        ).to(device)
    )

    binary_head = nn.Sequential(
        nn.Dropout(
            cfg[
                "HEAD_DROPOUT"
            ]
        ),

        nn.Linear(
            cfg["D_MODEL"],
            2,
        ),
    ).to(device)

    previous_binary_path = Path(
        cfg[
            "PREVIOUS_BINARY_CKPT"
        ]
    )

    reuse_stage1 = (
        bool(
            cfg[
                "REUSE_STAGE1_IF_EXISTS"
            ]
        )
        and previous_binary_path.exists()
    )

    # ========================================================
    # 直接加载已有 Stage 1
    # ========================================================
    if reuse_stage1:

        print(
            "[Stage 1] 加载已有模型：",
            previous_binary_path,
        )

        binary_checkpoint = torch.load(
            previous_binary_path,
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
            f"[Stage 1 Loaded] "
            f"Binary Score="
            f"{binary_checkpoint['binary_score']:.4f}, "
            f"Epoch="
            f"{binary_checkpoint['epoch']}, "
            f"Threshold="
            f"{binary_threshold:.4f}"
        )

    # ========================================================
    # 没有已有模型时重新训练 Stage 1
    # ========================================================
    else:

        binary_optimizer = (
            torch.optim.AdamW(
                list(
                    binary_backbone
                    .parameters()
                )
                + list(
                    binary_head
                    .parameters()
                ),

                lr=(
                    cfg[
                        "STAGE1_LR"
                    ]
                ),

                weight_decay=(
                    cfg[
                        "WEIGHT_DECAY"
                    ]
                ),
            )
        )

        binary_scheduler = (
            torch.optim.lr_scheduler
            .CosineAnnealingLR(
                binary_optimizer,

                T_max=(
                    cfg[
                        "STAGE1_EPOCHS"
                    ]
                ),

                eta_min=1e-6,
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
            cfg["STAGE1_EPOCHS"]
            + 1,
        ):

            start_time = time.time()

            (
                train_loss,
                optimizer_steps,
            ) = train_binary_epoch(
                train_loader,

                binary_backbone,
                binary_head,

                binary_optimizer,

                device,
                binary_scaler,
                use_amp,

                cfg[
                    "ACCUM_STEPS"
                ],
            )

            metrics = (
                evaluate_binary(
                    selection_loader,

                    binary_backbone,
                    binary_head,

                    device,
                    cfg,
                )
            )

            current_lr = (
                binary_optimizer
                .param_groups[0]["lr"]
            )

            if optimizer_steps > 0:

                binary_scheduler.step()

            if (
                metrics[
                    "BINARY_SCORE"
                ]
                > best_binary_score
                + 1e-9
            ):

                best_binary_score = (
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
                    binary_scheduler,

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
                binary_scheduler,

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
                f"{metrics['THRESHOLD']:.2f} | "

                f"LR "
                f"{current_lr:.8f} | "

                f"{time.time() - start_time:.1f}s"
            )

            if (
                bad_epochs
                >= cfg[
                    "STAGE1_PATIENCE"
                ]
            ):

                print(
                    "[Stage 1 Early Stop] "
                    f"Best Binary Score="
                    f"{best_binary_score:.4f}, "
                    f"Epoch="
                    f"{best_binary_epoch}"
                )

                break

        binary_checkpoint = (
            torch.load(
                best_binary_path,
                map_location=device,
            )
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

    # Stage 2 期间固定二分类模型
    binary_backbone.eval()
    binary_head.eval()

    for parameter in (
        list(
            binary_backbone.parameters()
        )
        + list(
            binary_head.parameters()
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
        "=" * 80
    )

    print(
        "STAGE 2: "
        "Crackle / Wheeze / Both"
    )

    print(
        "=" * 80
    )

    abnormal_backbone = (
        make_backbone(
            cfg
        ).to(device)
    )

    # 使用最佳二分类 Backbone 初始化
    abnormal_backbone.load_state_dict(
        binary_checkpoint[
            "backbone_state"
        ]
    )

    abnormal_head = nn.Sequential(
        nn.Dropout(
            cfg[
                "HEAD_DROPOUT"
            ]
        ),

        nn.Linear(
            cfg["D_MODEL"],
            3,
        ),
    ).to(device)

    # 前 3 个 Epoch 冻结 Backbone
    for parameter in (
        abnormal_backbone.parameters()
    ):

        parameter.requires_grad = (
            False
        )

    print(
        f"[Stage 2] 前 "
        f"{cfg['STAGE2_HEAD_ONLY_EPOCHS']} "
        "个 Epoch 只训练异常分类头。"
    )

    abnormal_optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params": (
                        abnormal_backbone
                        .parameters()
                    ),

                    "lr": (
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

                    "lr": (
                        cfg[
                            "STAGE2_HEAD_LR"
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
    )

    abnormal_scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            abnormal_optimizer,

            T_max=(
                cfg[
                    "STAGE2_EPOCHS"
                ]
            ),

            eta_min=5e-7,
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
        cfg["STAGE2_EPOCHS"]
        + 1,
    ):

        start_time = time.time()

        freeze_backbone = (
            epoch
            <= cfg[
                "STAGE2_HEAD_ONLY_EPOCHS"
            ]
        )

        # 第 4 个 Epoch 解冻 Backbone
        if (
            epoch
            == cfg[
                "STAGE2_HEAD_ONLY_EPOCHS"
            ]
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
        ) = train_abnormal_epoch(
            abnormal_train_loader,

            abnormal_backbone,
            abnormal_head,

            abnormal_optimizer,

            device,
            abnormal_scaler,
            use_amp,

            cfg[
                "ACCUM_STEPS"
            ],

            freeze_backbone,
        )

        abnormal_evaluation = (
            evaluate_abnormal(
                abnormal_selection_loader,

                abnormal_backbone,
                abnormal_head,

                device,
            )
        )

        cascade_evaluation = (
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

        backbone_lr = (
            abnormal_optimizer
            .param_groups[0]["lr"]
        )

        head_lr = (
            abnormal_optimizer
            .param_groups[1]["lr"]
        )

        if optimizer_steps > 0:

            abnormal_scheduler.step()

        if (
            cascade_evaluation[
                "FOUR_SCORE"
            ]
            > best_four_score
            + 1e-9
        ):

            best_four_score = (
                cascade_evaluation[
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
                abnormal_scheduler,

                cascade_evaluation,
                abnormal_evaluation,

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
            abnormal_scheduler,

            cascade_evaluation,
            abnormal_evaluation,

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
            f"{abnormal_evaluation['ABNORMAL_ACC']:.4f} | "

            f"Abn-F1 "
            f"{abnormal_evaluation['ABNORMAL_F1']:.4f} | "

            f"Abn-Recall "
            f"{np.round(abnormal_evaluation['ABNORMAL_RECALL'], 3).tolist()} | "

            f"4-Score "
            f"{cascade_evaluation['FOUR_SCORE']:.4f} | "

            f"4-SP "
            f"{cascade_evaluation['FOUR_SP']:.4f} | "

            f"4-SE "
            f"{cascade_evaluation['FOUR_SE']:.4f} | "

            f"Binary-Score "
            f"{cascade_evaluation['BINARY_SCORE']:.4f} | "

            f"B-LR "
            f"{backbone_lr:.8f} | "

            f"H-LR "
            f"{head_lr:.8f} | "

            f"{time.time() - start_time:.1f}s"
        )

        print(
            "    Recall[0,1,2,3]="
            f"{np.round(cascade_evaluation['RECALL'], 3).tolist()} | "
            f"PredCount="
            f"{cascade_evaluation['PRED_COUNTS'].tolist()}"
        )

        if (
            bad_epochs
            >= cfg[
                "STAGE2_PATIENCE"
            ]
        ):

            print(
                "[Stage 2 Early Stop] "
                f"Best 4-Class Score="
                f"{best_four_score:.4f}, "
                f"Epoch="
                f"{best_stage2_epoch}"
            )

            break

    # ========================================================
    # 加载最佳 Stage 2
    # ========================================================
    abnormal_checkpoint = (
        torch.load(
            best_abnormal_path,
            map_location=device,
        )
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

    # ========================================================
    # 最终测试
    # ========================================================
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
        (
            previous_binary_path
            if reuse_stage1
            else best_binary_path
        ),
    )

    print(
        "Best abnormal checkpoint:",
        best_abnormal_path,
    )


if __name__ == "__main__":
    main()