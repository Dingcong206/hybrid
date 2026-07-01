#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
    # 数据目录
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_ast_patch_tokens"
    ),

    # 训练参数
    "EPOCHS": 40,
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,

    "LR": 2e-5,
    "WEIGHT_DECAY": 1e-2,

    "NUM_WORKERS": 1,
    "SEED": 42,
    "DEVICE": "cuda",

    "AMP": True,
    "REQUIRE_MAMBA": True,

    "PATIENCE": 12,

    # 保存目录
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_serial_256_attnmax_sampler"
    ),

    # 模型参数
    "INPUT_DIM": 768,
    "D_MODEL": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,

    "DROPOUT": 0.15,
    "CLASSIFIER_DROPOUT": 0.20,

    # 使用温和过采样
    "USE_WEIGHTED_SAMPLER": True,
    "SAMPLER_POWER": 0.5,

    # 使用采样器后不再使用类别加权 CE
    "LABEL_SMOOTHING": 0.0,

    # 第一轮先关闭增强
    "SPEC_AUG": False,

    "MAX_MASK_T": 4,
    "MAX_MASK_F": 1,
    "NUM_MASKS": 1,

    # False：严格四分类
    # True：Normal / Abnormal 二分类式统计
    "TWO_CLS_EVAL": False,

    # 最佳模型组合指标
    "SELECTION_SCORE_WEIGHT": 0.5,
    "SELECTION_F1_WEIGHT": 0.5,
}


# ============================================================
# 导入模型
# ============================================================
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

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
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Patch 级 SpecAugment
# ============================================================
def apply_patch_augment(
    x,
    freq_patches,
    time_patches,
    max_mask_t,
    max_mask_f,
    num_masks,
):
    expected_tokens = (
        freq_patches
        * time_patches
    )

    if (
        x.ndim != 2
        or x.shape[0] != expected_tokens
    ):
        raise ValueError(
            f"SpecAugment 要求 "
            f"[{expected_tokens}, D]，"
            f"当前为 {tuple(x.shape)}。"
        )

    feature_dim = x.shape[1]

    x_aug = x.reshape(
        freq_patches,
        time_patches,
        feature_dim,
    ).clone()

    for _ in range(
        num_masks
    ):

        # 时间遮挡
        t_width = random.randint(
            0,
            min(
                max_mask_t,
                time_patches,
            ),
        )

        if t_width > 0:
            t_start = random.randint(
                0,
                time_patches - t_width,
            )

            x_aug[
                :,
                t_start:t_start + t_width,
                :
            ] = 0

        # 频率遮挡
        f_width = random.randint(
            0,
            min(
                max_mask_f,
                freq_patches,
            ),
        )

        if f_width > 0:
            f_start = random.randint(
                0,
                freq_patches - f_width,
            )

            x_aug[
                f_start:f_start + f_width,
                :,
                :
            ] = 0

    return x_aug.reshape(
        expected_tokens,
        feature_dim,
    )


# ============================================================
# Dataset
# ============================================================
class TokenNPY4ClsDataset(
    Dataset
):

    def __init__(
        self,
        csv_path,
        is_train,
        specaug,
        freq_patches,
        time_patches,
        input_dim,
        max_mask_t,
        max_mask_f,
        num_masks,
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
            - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{csv_path} 缺少列："
                f"{sorted(missing_columns)}；"
                f"当前列为："
                f"{self.df.columns.tolist()}。"
            )

        self.is_train = is_train
        self.specaug = specaug

        self.freq_patches = (
            freq_patches
        )

        self.time_patches = (
            time_patches
        )

        self.input_dim = (
            input_dim
        )

        self.expected_tokens = (
            freq_patches
            * time_patches
        )

        self.max_mask_t = (
            max_mask_t
        )

        self.max_mask_f = (
            max_mask_f
        )

        self.num_masks = (
            num_masks
        )

        self.labels = (
            self.df["label"]
            .astype(int)
            .to_numpy()
        )

        invalid_labels = np.unique(
            self.labels[
                (self.labels < 0)
                | (self.labels > 3)
            ]
        )

        if len(invalid_labels) > 0:
            raise ValueError(
                "标签必须是 0、1、2、3；"
                f"发现无效标签："
                f"{invalid_labels.tolist()}。"
            )

        self.class_counts_4 = (
            np.bincount(
                self.labels,
                minlength=4,
            )
        )

        print(
            f"[Dataset] Loaded "
            f"{len(self.df)} samples "
            f"from {csv_path}"
        )

        print(
            "[Dataset] class counts:",
            self.class_counts_4.tolist(),
        )

        print(
            f"[Dataset] train="
            f"{self.is_train}, "
            f"specaug={self.specaug}"
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
                    row["tokens_path"]
                )
            )
        )

        tokens_np = np.load(
            token_path
        )

        expected_shape = (
            self.expected_tokens,
            self.input_dim,
        )

        if (
            tuple(tokens_np.shape)
            != expected_shape
        ):
            raise ValueError(
                f"Token shape error："
                f"{token_path}\n"
                f"当前形状："
                f"{tuple(tokens_np.shape)}\n"
                f"要求形状："
                f"{expected_shape}"
            )

        x = torch.from_numpy(
            tokens_np
        ).float()

        if (
            self.is_train
            and self.specaug
        ):
            x = apply_patch_augment(
                x=x,

                freq_patches=(
                    self.freq_patches
                ),

                time_patches=(
                    self.time_patches
                ),

                max_mask_t=(
                    self.max_mask_t
                ),

                max_mask_f=(
                    self.max_mask_f
                ),

                num_masks=(
                    self.num_masks
                ),
            )

        y = torch.tensor(
            int(
                self.labels[index]
            ),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# 固定形状 Collate
# ============================================================
def collate_fixed(
    batch,
):
    xs, ys = zip(
        *batch
    )

    x_batch = torch.stack(
        xs,
        dim=0,
    )

    y_batch = torch.stack(
        ys,
        dim=0,
    ).view(-1)

    return (
        x_batch,
        y_batch,
    )


# ============================================================
# 温和 WeightedRandomSampler
# ============================================================
def build_weighted_sampler(
    labels,
    class_counts,
    power,
):
    # 类别采样权重：
    # 1 / count^0.5
    class_sample_weights = (
        1.0
        / np.power(
            np.maximum(
                class_counts.astype(
                    np.float64
                ),
                1.0,
            ),
            power,
        )
    )

    sample_weights = (
        class_sample_weights[
            labels
        ]
    )

    expected_mass = (
        class_counts
        * class_sample_weights
    )

    expected_ratio = (
        expected_mass
        / expected_mass.sum()
    )

    print(
        "[Sampler] class weights:",
        class_sample_weights.tolist(),
    )

    print(
        "[Sampler] expected sampled ratio:",
        expected_ratio.tolist(),
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),

        num_samples=len(
            sample_weights
        ),

        replacement=True,
    )


# ============================================================
# ICBHI Score
# ============================================================
def get_icbhi_score(
    hits,
    counts,
):
    eps = 1e-10

    sp = (
        100.0
        * hits[0]
        / (
            counts[0]
            + eps
        )
    )

    abnormal_hits = (
        hits[1]
        + hits[2]
        + hits[3]
    )

    abnormal_counts = (
        counts[1]
        + counts[2]
        + counts[3]
    )

    se = (
        100.0
        * abnormal_hits
        / (
            abnormal_counts
            + eps
        )
    )

    score = (
        sp + se
    ) / 2.0

    return (
        float(sp),
        float(se),
        float(score),
    )


# ============================================================
# 验证
# ============================================================
@torch.no_grad()
def evaluate(
    loader,
    backbone,
    classifier,
    device,
    two_cls_eval,
):
    backbone.eval()
    classifier.eval()

    hits = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    counts = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    all_true = []
    all_pred = []

    total_loss = 0.0
    total_num = 0

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    ).to(device)

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        feature = backbone(
            x
        )

        logits = classifier(
            feature
        )

        loss = criterion(
            logits,
            y,
        )

        total_loss += float(
            loss.item()
        )

        total_num += int(
            y.size(0)
        )

        pred = torch.argmax(
            logits,
            dim=1,
        )

        all_true.append(
            y.detach().cpu()
        )

        all_pred.append(
            pred.detach().cpu()
        )

        for index in range(
            y.size(0)
        ):
            gt = int(
                y[index].item()
            )

            pr = int(
                pred[index].item()
            )

            counts[gt] += 1.0

            if two_cls_eval:
                if (
                    gt == 0
                    and pr == 0
                ):
                    hits[gt] += 1.0

                elif (
                    gt != 0
                    and pr > 0
                ):
                    hits[gt] += 1.0

            elif pr == gt:
                hits[gt] += 1.0

    sp, se, score = (
        get_icbhi_score(
            hits,
            counts,
        )
    )

    y_true = torch.cat(
        all_true
    ).numpy()

    y_pred = torch.cat(
        all_pred
    ).numpy()

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

    predicted_counts = np.bincount(
        y_pred,
        minlength=4,
    )

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

    return {
        "SP": float(sp),
        "SE": float(se),
        "ICBHI": float(score),

        "ACC": float(
            accuracy
        ),

        "F1": float(
            macro_f1
        ),

        "LOSS": float(
            total_loss
            / max(
                total_num,
                1,
            )
        ),

        "class_recall": (
            class_recall
        ),

        "pred_counts": (
            predicted_counts
        ),

        "cm": confusion,
    }


# ============================================================
# AMP GradScaler
# ============================================================
def create_grad_scaler(
    use_amp,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )

    except (
        AttributeError,
        TypeError,
    ):
        return torch.cuda.amp.GradScaler(
            enabled=use_amp
        )


# ============================================================
# 训练一个 Epoch
# ============================================================
def train_one_epoch(
    loader,
    backbone,
    classifier,
    optimizer,
    device,
    use_amp,
    scaler,
    accum_steps,
    label_smoothing,
):
    backbone.train()
    classifier.train()

    # 使用 WeightedRandomSampler 后，
    # 不再额外加入类别权重
    criterion = nn.CrossEntropyLoss(
        label_smoothing=(
            label_smoothing
        )
    )

    parameters = (
        list(
            backbone.parameters()
        )
        + list(
            classifier.parameters()
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

    for batch_index, (x, y) in enumerate(
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

            feature = backbone(
                x
            )

            logits = classifier(
                feature
            )

            raw_loss = criterion(
                logits,
                y,
            )

            loss = (
                raw_loss
                / accum_steps
            )

        scaler.scale(
            loss
        ).backward()

        total_loss += float(
            raw_loss.detach().item()
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

            # Scale 未下降，表示 optimizer.step 未被跳过
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
# 形状检查
# ============================================================
@torch.no_grad()
def run_shape_test(
    loader,
    backbone,
    classifier,
    device,
    expected_dim,
):
    backbone.eval()
    classifier.eval()

    x, _ = next(
        iter(loader)
    )

    x = x[:1].to(
        device
    )

    feature = backbone(
        x
    )

    logits = classifier(
        feature
    )

    print(
        "[SHAPE TEST] input:",
        tuple(x.shape),
    )

    print(
        "[SHAPE TEST] feature:",
        tuple(feature.shape),
    )

    print(
        "[SHAPE TEST] logits:",
        tuple(logits.shape),
    )

    if (
        tuple(feature.shape)
        != (
            1,
            expected_dim,
        )
    ):
        raise RuntimeError(
            f"Backbone 输出为 "
            f"{tuple(feature.shape)}，"
            f"要求为 "
            f"(1, {expected_dim})。"
        )

    if (
        tuple(logits.shape)
        != (
            1,
            4,
        )
    ):
        raise RuntimeError(
            f"Classifier 输出为 "
            f"{tuple(logits.shape)}，"
            "要求为 (1, 4)。"
        )

    print(
        "[SHAPE TEST] Passed."
    )


# ============================================================
# 保存 Checkpoint
# ============================================================
def save_checkpoint(
    path,
    epoch,
    backbone,
    classifier,
    optimizer,
    scheduler,
    metrics,
    mix_metric,
    config,
):
    stored_metrics = {}

    for key, value in metrics.items():
        if isinstance(
            value,
            np.ndarray,
        ):
            stored_metrics[key] = (
                value.tolist()
            )
        else:
            stored_metrics[key] = (
                value
            )

    torch.save(
        {
            "epoch": epoch,

            "backbone_state": (
                backbone.state_dict()
            ),

            "classifier_state": (
                classifier.state_dict()
            ),

            "optimizer_state": (
                optimizer.state_dict()
            ),

            "scheduler_state": (
                scheduler.state_dict()
            ),

            "metrics": (
                stored_metrics
            ),

            "mix_metric": (
                mix_metric
            ),

            "config": deepcopy(
                config
            ),
        },
        path,
    )


# ============================================================
# 加载并评价 Checkpoint
# ============================================================
def report_checkpoint(
    path,
    name,
    backbone,
    classifier,
    loader,
    device,
    two_cls_eval,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    backbone.load_state_dict(
        checkpoint[
            "backbone_state"
        ]
    )

    classifier.load_state_dict(
        checkpoint[
            "classifier_state"
        ]
    )

    metrics = evaluate(
        loader=loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
        two_cls_eval=two_cls_eval,
    )

    print()
    print(
        f"[{name}]"
    )

    print(
        "Epoch:",
        checkpoint["epoch"],
    )

    print(
        f"Score: "
        f"{metrics['ICBHI']:.4f}"
    )

    print(
        f"SP: "
        f"{metrics['SP']:.4f}"
    )

    print(
        f"SE: "
        f"{metrics['SE']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{metrics['ACC']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{metrics['F1']:.4f}"
    )

    print(
        "Per-class recall:",
        np.round(
            metrics[
                "class_recall"
            ],
            4,
        ).tolist(),
    )

    print(
        "Predicted counts:",
        metrics[
            "pred_counts"
        ].tolist(),
    )

    print(
        "Confusion Matrix:"
    )

    print(
        metrics["cm"]
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
        f"[INFO] device: "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            "[INFO] CUDA device count:",
            torch.cuda.device_count(),
        )

        print(
            "[INFO] CUDA current device:",
            torch.cuda.current_device(),
        )

        print(
            "[INFO] CUDA device name:",
            torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
        )

    print(
        f"[INFO] HAS_MAMBA = "
        f"{HAS_MAMBA}"
    )

    if (
        cfg["REQUIRE_MAMBA"]
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm 导入失败，"
            "已停止正式训练。"
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

    # 优先使用独立验证集
    if val_csv.exists():

        eval_csv = val_csv

        print(
            "[INFO] 使用验证集：",
            eval_csv,
        )

    else:

        if not test_csv.exists():
            raise FileNotFoundError(
                test_csv
            )

        eval_csv = test_csv

        print(
            "[WARNING] 未找到 val_index.csv，"
            "当前每轮使用 test_index.csv 评价。"
        )

    # ========================================================
    # 保存路径
    # ========================================================
    os.makedirs(
        cfg["SAVE_DIR"],
        exist_ok=True,
    )

    best_score_path = os.path.join(
        cfg["SAVE_DIR"],
        "best_score.pth",
    )

    best_f1_path = os.path.join(
        cfg["SAVE_DIR"],
        "best_f1.pth",
    )

    best_mix_path = os.path.join(
        cfg["SAVE_DIR"],
        "best_mix.pth",
    )

    last_path = os.path.join(
        cfg["SAVE_DIR"],
        "last.pth",
    )

    # ========================================================
    # Dataset
    # ========================================================
    dataset_args = {
        "freq_patches": (
            cfg["FREQ_PATCHES"]
        ),

        "time_patches": (
            cfg["TIME_PATCHES"]
        ),

        "input_dim": (
            cfg["INPUT_DIM"]
        ),

        "max_mask_t": (
            cfg["MAX_MASK_T"]
        ),

        "max_mask_f": (
            cfg["MAX_MASK_F"]
        ),

        "num_masks": (
            cfg["NUM_MASKS"]
        ),
    }

    train_dataset = (
        TokenNPY4ClsDataset(
            csv_path=str(
                train_csv
            ),

            is_train=True,

            specaug=(
                cfg["SPEC_AUG"]
            ),

            **dataset_args,
        )
    )

    eval_dataset = (
        TokenNPY4ClsDataset(
            csv_path=str(
                eval_csv
            ),

            is_train=False,

            specaug=False,

            **dataset_args,
        )
    )

    # ========================================================
    # Weighted Sampler
    # ========================================================
    sampler = None
    shuffle = True

    if cfg[
        "USE_WEIGHTED_SAMPLER"
    ]:

        sampler = (
            build_weighted_sampler(
                labels=(
                    train_dataset.labels
                ),

                class_counts=(
                    train_dataset
                    .class_counts_4
                ),

                power=(
                    cfg[
                        "SAMPLER_POWER"
                    ]
                ),
            )
        )

        shuffle = False

    # ========================================================
    # DataLoader
    # ========================================================
    loader_args = {
        "batch_size": (
            cfg["BATCH_SIZE"]
        ),

        "num_workers": (
            cfg["NUM_WORKERS"]
        ),

        "pin_memory": (
            device.type == "cuda"
        ),

        "collate_fn": (
            collate_fixed
        ),

        "drop_last": False,

        "persistent_workers": (
            cfg["NUM_WORKERS"] > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        **loader_args,
    )

    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        **loader_args,
    )

    # ========================================================
    # Backbone
    # ========================================================
    backbone = TimeFrequencyEncoder(
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
    ).to(device)

    # ========================================================
    # Classifier
    # ========================================================
    classifier = nn.Sequential(
        nn.Dropout(
            cfg[
                "CLASSIFIER_DROPOUT"
            ]
        ),

        nn.Linear(
            cfg["D_MODEL"],
            4,
        ),
    ).to(device)

    print(
        "[MODEL] 768 -> 256 Projection"
    )

    print(
        "[MODEL] Time-Mamba "
        "-> Frequency-Attention"
    )

    print(
        "[MODEL] Attention Pooling "
        "+ Max Pooling "
        "-> Classifier"
    )

    # ========================================================
    # 形状检查
    # ========================================================
    run_shape_test(
        loader=train_loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
        expected_dim=(
            cfg["D_MODEL"]
        ),
    )

    parameters = (
        list(
            backbone.parameters()
        )
        + list(
            classifier.parameters()
        )
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in parameters
        if parameter.requires_grad
    )

    print(
        "[MODEL] Trainable parameters:",
        f"{trainable_count:,}",
    )

    # ========================================================
    # Optimizer
    # ========================================================
    optimizer = torch.optim.AdamW(
        parameters,

        lr=cfg["LR"],

        weight_decay=(
            cfg["WEIGHT_DECAY"]
        ),

        betas=(
            0.9,
            0.999,
        ),
    )

    # 每个 Epoch 更新一次
    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,

            T_max=(
                cfg["EPOCHS"]
            ),

            eta_min=2e-6,
        )
    )

    # ========================================================
    # AMP
    # ========================================================
    use_amp = (
        cfg["AMP"]
        and device.type == "cuda"
    )

    scaler = create_grad_scaler(
        use_amp
    )

    print(
        f"[INFO] AMP enabled: "
        f"{use_amp}"
    )

    # ========================================================
    # 最佳结果
    # ========================================================
    best_score = -1.0
    best_f1 = -1.0
    best_mix = -1.0

    best_score_epoch = -1
    best_f1_epoch = -1
    best_mix_epoch = -1

    bad_epochs = 0

    print()
    print(
        "=" * 72
    )

    print(
        "Start training: "
        "Serial 256D Time-Mamba "
        "-> Frequency-Attention"
    )

    print(
        "=" * 72
    )
    print()

    # ========================================================
    # Epoch Loop
    # ========================================================
    for epoch in range(
        1,
        cfg["EPOCHS"] + 1,
    ):

        start_time = time.time()

        train_loss, optimizer_steps = (
            train_one_epoch(
                loader=train_loader,
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
                accum_steps=(
                    cfg[
                        "ACCUM_STEPS"
                    ]
                ),
                label_smoothing=(
                    cfg[
                        "LABEL_SMOOTHING"
                    ]
                ),
            )
        )

        metrics = evaluate(
            loader=eval_loader,
            backbone=backbone,
            classifier=classifier,
            device=device,
            two_cls_eval=(
                cfg["TWO_CLS_EVAL"]
            ),
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        if optimizer_steps > 0:
            scheduler.step()

        score = float(
            metrics["ICBHI"]
        )

        macro_f1 = float(
            metrics["F1"]
        )

        mix_metric = (
            cfg[
                "SELECTION_SCORE_WEIGHT"
            ]
            * (
                score / 100.0
            )
            +
            cfg[
                "SELECTION_F1_WEIGHT"
            ]
            * macro_f1
        )

        markers = []

        # Best ICBHI Score
        if (
            score
            > best_score + 1e-9
        ):
            best_score = score
            best_score_epoch = epoch

            save_checkpoint(
                path=best_score_path,
                epoch=epoch,
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                mix_metric=mix_metric,
                config=cfg,
            )

            markers.append(
                "BEST-SCORE"
            )

        # Best Macro-F1
        if (
            macro_f1
            > best_f1 + 1e-9
        ):
            best_f1 = macro_f1
            best_f1_epoch = epoch

            save_checkpoint(
                path=best_f1_path,
                epoch=epoch,
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                mix_metric=mix_metric,
                config=cfg,
            )

            markers.append(
                "BEST-F1"
            )

        # Best Mixed Metric
        if (
            mix_metric
            > best_mix + 1e-9
        ):
            best_mix = mix_metric
            best_mix_epoch = epoch

            bad_epochs = 0

            save_checkpoint(
                path=best_mix_path,
                epoch=epoch,
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                mix_metric=mix_metric,
                config=cfg,
            )

            markers.append(
                "BEST-MIX"
            )

        else:
            bad_epochs += 1

        # Last checkpoint
        save_checkpoint(
            path=last_path,
            epoch=epoch,
            backbone=backbone,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            mix_metric=mix_metric,
            config=cfg,
        )

        recalls = np.round(
            metrics[
                "class_recall"
            ],
            3,
        ).tolist()

        predicted_counts = (
            metrics[
                "pred_counts"
            ].tolist()
        )

        marker_text = (
            ",".join(markers)
            if markers
            else "-"
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        print(
            f"[{marker_text}] "
            f"Epoch {epoch:03d}/"
            f"{cfg['EPOCHS']} | "

            f"train "
            f"{train_loss:.4f} | "

            f"val "
            f"{metrics['LOSS']:.4f} | "

            f"Score "
            f"{score:.4f} | "

            f"SP "
            f"{metrics['SP']:.4f} | "

            f"SE "
            f"{metrics['SE']:.4f} | "

            f"ACC "
            f"{metrics['ACC']:.4f} | "

            f"F1 "
            f"{macro_f1:.4f} | "

            f"Mix "
            f"{mix_metric:.4f} | "

            f"LR "
            f"{current_lr:.8f} | "

            f"{elapsed_time:.1f}s"
        )

        print(
            "    Recall[0,1,2,3]="
            f"{recalls} | "
            "PredCount="
            f"{predicted_counts}"
        )

        # Early stopping 根据组合指标
        if (
            bad_epochs
            >= cfg["PATIENCE"]
        ):
            print(
                "[EARLY STOP] "
                f"Mix 连续 "
                f"{cfg['PATIENCE']} "
                "个 Epoch 没有提升。"
            )

            break

    # ========================================================
    # 训练结束
    # ========================================================
    print()
    print(
        "=" * 72
    )

    print(
        f"Best Score: "
        f"{best_score:.4f} "
        f"at Epoch "
        f"{best_score_epoch}"
    )

    print(
        f"Best Macro-F1: "
        f"{best_f1:.4f} "
        f"at Epoch "
        f"{best_f1_epoch}"
    )

    print(
        f"Best Mix: "
        f"{best_mix:.4f} "
        f"at Epoch "
        f"{best_mix_epoch}"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 分别评价三个最佳模型
    # ========================================================
    report_checkpoint(
        path=best_mix_path,
        name="BEST MIX",
        backbone=backbone,
        classifier=classifier,
        loader=eval_loader,
        device=device,
        two_cls_eval=(
            cfg["TWO_CLS_EVAL"]
        ),
    )

    report_checkpoint(
        path=best_score_path,
        name="BEST SCORE",
        backbone=backbone,
        classifier=classifier,
        loader=eval_loader,
        device=device,
        two_cls_eval=(
            cfg["TWO_CLS_EVAL"]
        ),
    )

    report_checkpoint(
        path=best_f1_path,
        name="BEST F1",
        backbone=backbone,
        classifier=classifier,
        loader=eval_loader,
        device=device,
        two_cls_eval=(
            cfg["TWO_CLS_EVAL"]
        ),
    )


if __name__ == "__main__":
    main()