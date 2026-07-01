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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset


CONFIG = {
    "ROOT": "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",

    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_auxiliary_hierarchical_v2"
    ),

    "INIT_FROM_BINARY": True,

    "BINARY_CKPT": (
        "/data/dingcong/hybrid/"
        "checkpoints_two_stage_cascade/"
        "best_binary.pth"
    ),

    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 1,

    "EPOCHS": 40,

    # 前 15 个 Epoch 不允许早停
    "MIN_EPOCHS": 15,

    # 第 15 个 Epoch 以后连续 10 轮无提升才停止
    "PATIENCE": 10,

    # 只冻结前两个 Epoch 的 Backbone
    "BACKBONE_FREEZE_EPOCHS": 2,

    # 异常类别总体敏感度低于 20 的模型不作为最佳模型
    "MIN_VALID_FOUR_SE": 20.0,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    "INPUT_DIM": 768,
    "D_MODEL": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,
    "DROPOUT": 0.15,

    "HEAD_DROPOUT": 0.20,

    "BACKBONE_LR": 2e-6,
    "FOUR_HEAD_LR": 2e-5,
    "ABNORMAL_HEAD_LR": 1e-5,

    "MIN_LR": 5e-7,

    # 主任务四分类损失
    "FOUR_LOSS_WEIGHT": 1.0,

    # 二分类辅助损失
    "BINARY_LOSS_WEIGHT": 0.20,

    # 异常三分类辅助损失
    "ABNORMAL_LOSS_WEIGHT": 0.30,

    "ABNORMAL_WEIGHT_POWER": 0.5,
}


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


def set_seed(
    seed: int,
) -> None:

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
            enabled=enabled
        )


def safe_torch_load(
    path: Path,
    device: torch.device,
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


def make_backbone(
    cfg,
):

    return TimeFrequencyEncoder(
        input_dim=int(
            cfg["INPUT_DIM"]
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
    )


class AuxiliaryHierarchicalClassifier(
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

        # 最终四分类预测头
        self.four_head = nn.Sequential(
            nn.Dropout(
                float(
                    cfg["HEAD_DROPOUT"]
                )
            ),

            nn.Linear(
                int(
                    cfg["D_MODEL"]
                ),
                4,
            ),
        )

        # 二分类辅助头
        self.binary_head = nn.Sequential(
            nn.Dropout(
                float(
                    cfg["HEAD_DROPOUT"]
                )
            ),

            nn.Linear(
                int(
                    cfg["D_MODEL"]
                ),
                2,
            ),
        )

        # 异常三分类辅助头
        self.abnormal_head = nn.Sequential(
            nn.Dropout(
                float(
                    cfg["HEAD_DROPOUT"]
                )
            ),

            nn.Linear(
                int(
                    cfg["D_MODEL"]
                ),
                3,
            ),
        )

    def forward(
        self,
        x,
    ):

        feature = self.backbone(
            x
        )

        return {
            "feature": feature,

            "four_logits": self.four_head(
                feature
            ),

            "binary_logits": self.binary_head(
                feature
            ),

            "abnormal_logits": self.abnormal_head(
                feature
            ),
        }


class TokenDataset(
    Dataset
):

    def __init__(
        self,
        csv_path,
        cfg,
    ):

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
                f"{sorted(missing_columns)}"
            )

        self.df["label"] = (
            self.df["label"]
            .astype(int)
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
                "标签必须为 0、1、2、3，"
                f"发现：{invalid_labels.tolist()}"
            )

        self.expected_shape = (
            int(
                cfg["FREQ_PATCHES"]
            )
            *
            int(
                cfg["TIME_PATCHES"]
            ),

            int(
                cfg["INPUT_DIM"]
            ),
        )

        self.class_counts = np.bincount(
            self.labels,
            minlength=4,
        )

        print(
            f"[Dataset] "
            f"samples={len(self.df)} | "
            f"counts={self.class_counts.tolist()} | "
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
        raw_path: str,
    ) -> Path:

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
                row["tokens_path"]
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
                f"当前={tuple(tokens.shape)}，"
                f"要求={self.expected_shape}"
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
            device.type
            == "cuda"
        ),

        persistent_workers=(
            workers > 0
        ),

        drop_last=False,

        collate_fn=collate_fixed,
    )


def build_abnormal_class_weights(
    class_counts,
    power,
):

    abnormal_counts = (
        class_counts[
            1:4
        ]
        .astype(
            np.float64
        )
    )

    weights = (
        1.0
        /
        np.power(
            np.maximum(
                abnormal_counts,
                1.0,
            ),
            float(
                power
            ),
        )
    )

    weights = (
        weights
        /
        weights.mean()
    )

    print(
        "[Loss] abnormal counts:",
        abnormal_counts.tolist(),
    )

    print(
        "[Loss] abnormal weights:",
        np.round(
            weights,
            6,
        ).tolist(),
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def build_four_class_weights(
    class_counts,
):

    class_counts = np.asarray(
        class_counts,
        dtype=np.float64,
    )

    abnormal_counts = (
        class_counts[
            1:4
        ]
    )

    raw_weights = (
        1.0
        /
        np.sqrt(
            np.maximum(
                abnormal_counts,
                1.0,
            )
        )
    )

    scale = (
        abnormal_counts.sum()
        /
        np.sum(
            abnormal_counts
            * raw_weights
        )
    )

    abnormal_weights = (
        raw_weights
        * scale
    )

    weights = np.asarray(
        [
            1.0,
            abnormal_weights[0],
            abnormal_weights[1],
            abnormal_weights[2],
        ],

        dtype=np.float32,
    )

    print(
        "[Loss] four-class counts:",
        class_counts
        .astype(int)
        .tolist(),
    )

    print(
        "[Loss] four-class weights:",
        np.round(
            weights,
            6,
        ).tolist(),
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def calculate_binary_metrics(
    y_true_four,
    y_pred_binary,
    prefix,
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

    sp = (
        100.0
        * float(
            cm[0, 0]
        )
        / max(
            normal_total,
            1.0,
        )
    )

    se = (
        100.0
        * float(
            cm[1, 1]
        )
        / max(
            abnormal_total,
            1.0,
        )
    )

    return {
        f"{prefix}_SP": sp,

        f"{prefix}_SE": se,

        f"{prefix}_SCORE": (
            sp + se
        ) / 2.0,

        f"{prefix}_CM": cm,
    }


def calculate_four_class_metrics(
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

    correct_abnormal = float(
        cm[1, 1]
        + cm[2, 2]
        + cm[3, 3]
    )

    four_se = (
        100.0
        * correct_abnormal
        / max(
            abnormal_total,
            1.0,
        )
    )

    four_score = (
        four_sp
        + four_se
    ) / 2.0

    return {
        "FOUR_SP": four_sp,

        "FOUR_SE": four_se,

        "FOUR_SCORE": four_score,

        "ACC": (
            accuracy_score(
                y_true,
                y_pred,
            )
            * 100.0
        ),

        "MACRO_F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
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

        "PRED_COUNTS": np.bincount(
            y_pred,
            minlength=4,
        ),

        "FOUR_CM": cm,
    }


def calculate_joint_loss(
    outputs,
    labels,
    four_class_weights,
    abnormal_class_weights,
    cfg,
):

    four_logits = outputs[
        "four_logits"
    ]

    binary_logits = outputs[
        "binary_logits"
    ]

    abnormal_logits = outputs[
        "abnormal_logits"
    ]

    binary_target = (
        labels > 0
    ).long()

    abnormal_mask = (
        labels > 0
    )

    # 四分类主损失
    four_loss = F.cross_entropy(
        four_logits,
        labels,
        weight=four_class_weights,
    )

    # 二分类辅助损失
    binary_loss = F.cross_entropy(
        binary_logits,
        binary_target,
    )

    # 异常三分类辅助损失
    if abnormal_mask.any():

        abnormal_target = (
            labels[
                abnormal_mask
            ]
            - 1
        )

        abnormal_loss = F.cross_entropy(
            abnormal_logits[
                abnormal_mask
            ],

            abnormal_target,

            weight=abnormal_class_weights,
        )

    else:

        abnormal_loss = torch.zeros(
            (),
            dtype=four_loss.dtype,
            device=four_loss.device,
        )

    total_loss = (
        float(
            cfg["FOUR_LOSS_WEIGHT"]
        )
        * four_loss

        +

        float(
            cfg["BINARY_LOSS_WEIGHT"]
        )
        * binary_loss

        +

        float(
            cfg["ABNORMAL_LOSS_WEIGHT"]
        )
        * abnormal_loss
    )

    return {
        "total": total_loss,

        "four": four_loss,

        "binary": binary_loss,

        "abnormal": abnormal_loss,
    }


def train_one_epoch(
    loader,
    model,
    optimizer,
    device,
    scaler,
    use_amp,
    four_class_weights,
    abnormal_class_weights,
    cfg,
    freeze_backbone,
):

    model.train()

    # 二分类头始终固定，关闭其中的 Dropout
    model.binary_head.eval()

    if freeze_backbone:

        model.backbone.eval()

    else:

        model.backbone.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulation_steps = int(
        cfg["ACCUM_STEPS"]
    )

    number_of_batches = len(
        loader
    )

    loss_sum = {
        "total": 0.0,
        "four": 0.0,
        "binary": 0.0,
        "abnormal": 0.0,
    }

    optimizer_steps = 0

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

            outputs = model(
                x
            )

            losses = calculate_joint_loss(
                outputs,
                y,
                four_class_weights,
                abnormal_class_weights,
                cfg,
            )

            scaled_loss = (
                losses["total"]
                / accumulation_steps
            )

        scaler.scale(
            scaled_loss
        ).backward()

        for key in loss_sum:

            loss_sum[key] += float(
                losses[key]
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

                max_norm=float(
                    cfg["GRAD_CLIP"]
                ),
            )

            old_scale = (
                scaler.get_scale()
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            if (
                scaler.get_scale()
                >= old_scale
            ):

                optimizer_steps += 1

            optimizer.zero_grad(
                set_to_none=True
            )

    divisor = max(
        number_of_batches,
        1,
    )

    return {
        "TOTAL_LOSS": (
            loss_sum["total"]
            / divisor
        ),

        "FOUR_LOSS": (
            loss_sum["four"]
            / divisor
        ),

        "BINARY_LOSS": (
            loss_sum["binary"]
            / divisor
        ),

        "ABNORMAL_LOSS": (
            loss_sum["abnormal"]
            / divisor
        ),

        "OPTIMIZER_STEPS": (
            optimizer_steps
        ),
    }


@torch.no_grad()
def evaluate(
    loader,
    model,
    device,
    four_class_weights,
    abnormal_class_weights,
    cfg,
):

    model.eval()

    all_true = []
    all_four_prediction = []
    all_binary_prediction = []

    loss_sum = {
        "total": 0.0,
        "four": 0.0,
        "binary": 0.0,
        "abnormal": 0.0,
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

        outputs = model(
            x
        )

        losses = calculate_joint_loss(
            outputs,
            y,
            four_class_weights,
            abnormal_class_weights,
            cfg,
        )

        four_prediction = (
            outputs[
                "four_logits"
            ]
            .argmax(
                dim=1
            )
        )

        binary_prediction = (
            outputs[
                "binary_logits"
            ]
            .argmax(
                dim=1
            )
        )

        all_true.append(
            y.detach().cpu()
        )

        all_four_prediction.append(
            four_prediction
            .detach()
            .cpu()
        )

        all_binary_prediction.append(
            binary_prediction
            .detach()
            .cpu()
        )

        for key in loss_sum:

            loss_sum[key] += float(
                losses[key].item()
            )

    y_true = (
        torch.cat(
            all_true
        )
        .numpy()
    )

    y_pred_four = (
        torch.cat(
            all_four_prediction
        )
        .numpy()
    )

    y_pred_gate_binary = (
        torch.cat(
            all_binary_prediction
        )
        .numpy()
    )

    y_pred_final_binary = (
        y_pred_four > 0
    ).astype(
        np.int64
    )

    metrics = {}

    metrics.update(
        calculate_four_class_metrics(
            y_true,
            y_pred_four,
        )
    )

    metrics.update(
        calculate_binary_metrics(
            y_true,
            y_pred_final_binary,
            prefix="FINAL_BINARY",
        )
    )

    metrics.update(
        calculate_binary_metrics(
            y_true,
            y_pred_gate_binary,
            prefix="GATE_BINARY",
        )
    )

    divisor = max(
        len(
            loader
        ),
        1,
    )

    metrics.update(
        {
            "TOTAL_LOSS": (
                loss_sum["total"]
                / divisor
            ),

            "FOUR_LOSS": (
                loss_sum["four"]
                / divisor
            ),

            "BINARY_LOSS": (
                loss_sum["binary"]
                / divisor
            ),

            "ABNORMAL_LOSS": (
                loss_sum["abnormal"]
                / divisor
            ),
        }
    )

    return metrics


@torch.no_grad()
def run_shape_test(
    loader,
    model,
    device,
    cfg,
):

    model.eval()

    x, _ = next(
        iter(
            loader
        )
    )

    x = x[
        :1
    ].to(
        device
    )

    outputs = model(
        x
    )

    expected_shapes = {
        "feature": (
            1,
            int(
                cfg["D_MODEL"]
            ),
        ),

        "four_logits": (
            1,
            4,
        ),

        "binary_logits": (
            1,
            2,
        ),

        "abnormal_logits": (
            1,
            3,
        ),
    }

    print(
        "[Shape] input:",
        tuple(
            x.shape
        ),
    )

    for key, expected_shape in (
        expected_shapes.items()
    ):

        current_shape = tuple(
            outputs[
                key
            ].shape
        )

        print(
            f"[Shape] {key}:",
            current_shape,
        )

        if current_shape != expected_shape:

            raise RuntimeError(
                f"{key} shape 错误："
                f"{current_shape} != "
                f"{expected_shape}"
            )

    print(
        "[Shape] Passed."
    )


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

            result[key] = value

    return result


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    metrics,
    four_class_weights,
    abnormal_class_weights,
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

            "backbone_state": (
                model.backbone
                .state_dict()
            ),

            "four_head_state": (
                model.four_head
                .state_dict()
            ),

            "binary_head_state": (
                model.binary_head
                .state_dict()
            ),

            "abnormal_head_state": (
                model.abnormal_head
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

            "four_sp": float(
                metrics[
                    "FOUR_SP"
                ]
            ),

            "four_se": float(
                metrics[
                    "FOUR_SE"
                ]
            ),

            "macro_f1": float(
                metrics[
                    "MACRO_F1"
                ]
            ),

            "four_class_weights": (
                four_class_weights
                .detach()
                .cpu()
                .tolist()
            ),

            "abnormal_class_weights": (
                abnormal_class_weights
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


def print_final_results(
    metrics,
):

    print()

    print(
        "=" * 80
    )

    print(
        "[FINAL AUXILIARY HIERARCHICAL TEST]"
    )

    print(
        "=" * 80
    )

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
        f"{metrics['MACRO_F1']:.4f}"
    )

    print()

    print(
        f"Final Binary Score: "
        f"{metrics['FINAL_BINARY_SCORE']:.4f}"
    )

    print(
        f"Final Binary SP: "
        f"{metrics['FINAL_BINARY_SP']:.4f}"
    )

    print(
        f"Final Binary SE: "
        f"{metrics['FINAL_BINARY_SE']:.4f}"
    )

    print()

    print(
        f"Fixed Gate Score: "
        f"{metrics['GATE_BINARY_SCORE']:.4f}"
    )

    print(
        f"Fixed Gate SP: "
        f"{metrics['GATE_BINARY_SP']:.4f}"
    )

    print(
        f"Fixed Gate SE: "
        f"{metrics['GATE_BINARY_SE']:.4f}"
    )

    print()

    print(
        "Recall[Normal,Crackle,Wheeze,Both]:",

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
        "Four-class confusion matrix:"
    )

    print(
        metrics[
            "FOUR_CM"
        ]
    )

    print()

    print(
        "Final binary confusion matrix:"
    )

    print(
        metrics[
            "FINAL_BINARY_CM"
        ]
    )

    print()

    print(
        "Fixed binary-gate confusion matrix:"
    )

    print(
        metrics[
            "GATE_BINARY_CM"
        ]
    )


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
        bool(
            cfg["REQUIRE_MAMBA"]
        )
        and
        not HAS_MAMBA
    ):

        raise RuntimeError(
            "mamba_ssm 导入失败。"
        )

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
            "选择 checkpoint。"
        )

        print(
            "[WARNING] 正式实验应建立独立验证集，"
            "否则存在测试集泄漏。"
        )

    save_dir = Path(
        cfg["SAVE_DIR"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        save_dir
        / "best_auxiliary_hierarchical.pth"
    )

    fallback_checkpoint_path = (
        save_dir
        / "best_fallback_f1.pth"
    )

    last_checkpoint_path = (
        save_dir
        / "last_auxiliary_hierarchical.pth"
    )

    train_dataset = TokenDataset(
        train_csv,
        cfg,
    )

    selection_dataset = TokenDataset(
        selection_csv,
        cfg,
    )

    test_dataset = TokenDataset(
        test_csv,
        cfg,
    )

    train_loader = make_loader(
        train_dataset,
        cfg,
        device,
        shuffle=True,
    )

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

    model = (
        AuxiliaryHierarchicalClassifier(
            cfg
        )
        .to(
            device
        )
    )

    binary_checkpoint_path = Path(
        cfg[
            "BINARY_CKPT"
        ]
    )

    if (
        bool(
            cfg[
                "INIT_FROM_BINARY"
            ]
        )
        and
        binary_checkpoint_path.exists()
    ):

        binary_checkpoint = safe_torch_load(
            binary_checkpoint_path,
            device,
        )

        model.backbone.load_state_dict(
            binary_checkpoint[
                "backbone_state"
            ]
        )

        model.binary_head.load_state_dict(
            binary_checkpoint[
                "binary_head_state"
            ]
        )

        print(
            "[INIT] Loaded previous "
            "binary checkpoint:",
            binary_checkpoint_path,
        )

        print(
            f"[INIT] Previous Binary Score="
            f"{binary_checkpoint.get('binary_score', float('nan')):.4f}"
        )

    else:

        print(
            "[INIT] Binary checkpoint "
            "not loaded."
        )

    # 二分类头始终冻结
    for parameter in (
        model.binary_head.parameters()
    ):

        parameter.requires_grad = False

    # Backbone 初始冻结
    for parameter in (
        model.backbone.parameters()
    ):

        parameter.requires_grad = False

    run_shape_test(
        train_loader,
        model,
        device,
        cfg,
    )

    four_class_weights = (
        build_four_class_weights(
            train_dataset.class_counts
        )
        .to(
            device
        )
    )

    abnormal_class_weights = (
        build_abnormal_class_weights(
            train_dataset.class_counts,

            cfg[
                "ABNORMAL_WEIGHT_POWER"
            ],
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.backbone
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "BACKBONE_LR"
                    ]
                ),
            },

            {
                "params": (
                    model.four_head
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "FOUR_HEAD_LR"
                    ]
                ),
            },

            {
                "params": (
                    model.abnormal_head
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "ABNORMAL_HEAD_LR"
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
                    "EPOCHS"
                ]
            ),

            eta_min=float(
                cfg[
                    "MIN_LR"
                ]
            ),
        )
    )

    use_amp = bool(
        cfg["AMP"]
        and
        device.type == "cuda"
    )

    scaler = make_scaler(
        use_amp
    )

    best_four_score = float(
        "-inf"
    )

    best_four_f1 = float(
        "-inf"
    )

    best_fallback_f1 = float(
        "-inf"
    )

    best_epoch = -1

    bad_epochs = 0

    has_valid_best = False

    print()

    print(
        "=" * 90
    )

    print(
        "AUXILIARY HIERARCHICAL "
        "JOINT TRAINING"
    )

    print(
        "=" * 90
    )

    for epoch in range(
        1,
        int(
            cfg["EPOCHS"]
        )
        + 1,
    ):

        start_time = time.time()

        freeze_backbone = (
            epoch
            <= int(
                cfg[
                    "BACKBONE_FREEZE_EPOCHS"
                ]
            )
        )

        for parameter in (
            model.backbone.parameters()
        ):

            parameter.requires_grad = (
                not freeze_backbone
            )

        if (
            epoch
            == int(
                cfg[
                    "BACKBONE_FREEZE_EPOCHS"
                ]
            )
            + 1
        ):

            print(
                "[Training] Backbone 已解冻，"
                "开始联合微调。"
            )

        train_metrics = train_one_epoch(
            train_loader,
            model,
            optimizer,
            device,
            scaler,
            use_amp,
            four_class_weights,
            abnormal_class_weights,
            cfg,
            freeze_backbone,
        )

        validation_metrics = evaluate(
            selection_loader,
            model,
            device,
            four_class_weights,
            abnormal_class_weights,
            cfg,
        )

        current_learning_rates = [
            group["lr"]
            for group
            in optimizer.param_groups
        ]

        if (
            train_metrics[
                "OPTIMIZER_STEPS"
            ]
            > 0
        ):

            scheduler.step()

        current_four_score = float(
            validation_metrics[
                "FOUR_SCORE"
            ]
        )

        current_four_se = float(
            validation_metrics[
                "FOUR_SE"
            ]
        )

        current_macro_f1 = float(
            validation_metrics[
                "MACRO_F1"
            ]
        )

        # 始终保存最后一轮
        save_checkpoint(
            last_checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            validation_metrics,
            four_class_weights,
            abnormal_class_weights,
            cfg,
        )

        # 始终保存 Macro-F1 最好的备用模型
        if (
            current_macro_f1
            > best_fallback_f1
            + 1e-9
        ):

            best_fallback_f1 = (
                current_macro_f1
            )

            save_checkpoint(
                fallback_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                validation_metrics,
                four_class_weights,
                abnormal_class_weights,
                cfg,
            )

        valid_candidate = (
            epoch
            > int(
                cfg[
                    "BACKBONE_FREEZE_EPOCHS"
                ]
            )
            and
            current_four_se
            >= float(
                cfg[
                    "MIN_VALID_FOUR_SE"
                ]
            )
        )

        improved = False

        if valid_candidate:

            if (
                current_four_score
                > best_four_score
                + 1e-9
            ):

                improved = True

            elif (
                abs(
                    current_four_score
                    - best_four_score
                )
                <= 1e-9

                and

                current_macro_f1
                > best_four_f1
                + 1e-9
            ):

                improved = True

        if improved:

            best_four_score = (
                current_four_score
            )

            best_four_f1 = (
                current_macro_f1
            )

            best_epoch = (
                epoch
            )

            has_valid_best = True

            bad_epochs = 0

            marker = (
                "BEST-4SCORE"
            )

            save_checkpoint(
                best_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                validation_metrics,
                four_class_weights,
                abnormal_class_weights,
                cfg,
            )

        else:

            marker = "-"

            # 前 15 个 Epoch 不累计 bad_epochs
            # 没有出现有效最佳模型时也不累计
            if (
                epoch
                >= int(
                    cfg[
                        "MIN_EPOCHS"
                    ]
                )

                and

                has_valid_best
            ):

                bad_epochs += 1

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[{marker}] "
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['EPOCHS']} | "

            f"Mode "
            f"{'Frozen' if freeze_backbone else 'Fine-Tune'} | "

            f"Train "
            f"{train_metrics['TOTAL_LOSS']:.4f} | "

            f"FourLoss "
            f"{train_metrics['FOUR_LOSS']:.4f} | "

            f"BinLoss "
            f"{train_metrics['BINARY_LOSS']:.4f} | "

            f"AbnLoss "
            f"{train_metrics['ABNORMAL_LOSS']:.4f} | "

            f"4-Score "
            f"{current_four_score:.4f} | "

            f"4-SP "
            f"{validation_metrics['FOUR_SP']:.4f} | "

            f"4-SE "
            f"{current_four_se:.4f} | "

            f"F1 "
            f"{current_macro_f1:.4f} | "

            f"Valid "
            f"{valid_candidate} | "

            f"Bad "
            f"{bad_epochs}/"
            f"{cfg['PATIENCE']} | "

            f"LR "
            f"{current_learning_rates[0]:.8f}/"
            f"{current_learning_rates[1]:.8f}/"
            f"{current_learning_rates[2]:.8f} | "

            f"{elapsed:.1f}s"
        )

        print(
            "    Recall[0,1,2,3]="
            f"{np.round(validation_metrics['RECALL'], 3).tolist()} | "

            f"PredCount="
            f"{validation_metrics['PRED_COUNTS'].tolist()} | "

            f"Final-Bin="
            f"{validation_metrics['FINAL_BINARY_SCORE']:.4f} | "

            f"Gate-Bin="
            f"{validation_metrics['GATE_BINARY_SCORE']:.4f}"
        )

        should_early_stop = (
            epoch
            >= int(
                cfg[
                    "MIN_EPOCHS"
                ]
            )

            and

            has_valid_best

            and

            bad_epochs
            >= int(
                cfg[
                    "PATIENCE"
                ]
            )
        )

        if should_early_stop:

            print(
                "[Early Stop] "
                f"Epoch={epoch}, "
                f"Best Epoch={best_epoch}, "
                f"Best Score="
                f"{best_four_score:.4f}, "
                f"Bad Epochs="
                f"{bad_epochs}"
            )

            break

    if best_checkpoint_path.exists():

        selected_checkpoint_path = (
            best_checkpoint_path
        )

        print(
            "[Checkpoint] 使用满足 "
            "FOUR_SE 条件的最佳模型。"
        )

    elif fallback_checkpoint_path.exists():

        selected_checkpoint_path = (
            fallback_checkpoint_path
        )

        print(
            "[Checkpoint] 没有有效主模型，"
            "使用 Macro-F1 最佳模型。"
        )

    else:

        selected_checkpoint_path = (
            last_checkpoint_path
        )

        print(
            "[Checkpoint] 使用最后一轮模型。"
        )

    selected_checkpoint = safe_torch_load(
        selected_checkpoint_path,
        device,
    )

    model.load_state_dict(
        selected_checkpoint[
            "model_state"
        ]
    )

    print(
        "[Training Completed] "
        f"Selected Epoch="
        f"{selected_checkpoint['epoch']}, "
        f"Score="
        f"{selected_checkpoint['four_score']:.4f}, "
        f"SE="
        f"{selected_checkpoint['four_se']:.4f}, "
        f"F1="
        f"{selected_checkpoint['macro_f1']:.4f}"
    )

    final_metrics = evaluate(
        test_loader,
        model,
        device,
        four_class_weights,
        abnormal_class_weights,
        cfg,
    )

    print_final_results(
        final_metrics
    )

    print()

    print(
        "Selected checkpoint:",
        selected_checkpoint_path,
    )

    print(
        "Best valid checkpoint:",
        best_checkpoint_path,
    )

    print(
        "Fallback checkpoint:",
        fallback_checkpoint_path,
    )

    print(
        "Last checkpoint:",
        last_checkpoint_path,
    )


if __name__ == "__main__":

    main()