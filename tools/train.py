#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Soft hierarchical joint training for ICBHI.

Shared backbone:
    AST tokens
        -> Time-Mamba
        -> Frequency-Attention
        -> shared feature

Two heads:
    Binary head:
        Normal / Abnormal

    Abnormal head:
        Crackle / Wheeze / Both

Final four-class probabilities:
    P(Normal)  = P(Normal)

    P(Crackle) = P(Abnormal) * P(Crackle | Abnormal)
    P(Wheeze)  = P(Abnormal) * P(Wheeze  | Abnormal)
    P(Both)    = P(Abnormal) * P(Both    | Abnormal)

There is no hard binary threshold during final four-class prediction.
"""

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
# 1. Configuration
# ============================================================
CONFIG = {
    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_ast_patch_tokens"
    ),

    # --------------------------------------------------------
    # Save directory
    # --------------------------------------------------------
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_soft_hierarchical"
    ),

    # --------------------------------------------------------
    # Warm start from the previous binary model
    # --------------------------------------------------------
    "INIT_FROM_BINARY": True,

    "BINARY_CKPT": (
        "/data/dingcong/hybrid/"
        "checkpoints_two_stage_cascade/"
        "best_binary.pth"
    ),

    # --------------------------------------------------------
    # General training
    # --------------------------------------------------------
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 1,

    "EPOCHS": 40,
    "PATIENCE": 12,

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
    # Classification heads
    # --------------------------------------------------------
    "HEAD_DROPOUT": 0.20,

    # --------------------------------------------------------
    # Learning rates
    # --------------------------------------------------------
    "BACKBONE_LR": 5e-6,
    "BINARY_HEAD_LR": 1e-5,
    "ABNORMAL_HEAD_LR": 2e-5,

    "MIN_LR": 5e-7,

    # --------------------------------------------------------
    # Joint loss coefficients
    #
    # total_loss =
    #     FOUR_LOSS_WEIGHT * four_loss
    #   + BINARY_LOSS_WEIGHT * binary_loss
    #   + ABNORMAL_LOSS_WEIGHT * abnormal_loss
    # --------------------------------------------------------
    "FOUR_LOSS_WEIGHT": 1.0,
    "BINARY_LOSS_WEIGHT": 0.30,
    "ABNORMAL_LOSS_WEIGHT": 0.50,

    # --------------------------------------------------------
    # Mild abnormal class weights
    #
    # weight = 1 / count^power
    # --------------------------------------------------------
    "ABNORMAL_WEIGHT_POWER": 0.5,

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------
    "GRAD_CLIP": 2.0,
}


# ============================================================
# 2. Import model
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
# 3. Random seed
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
# 4. AMP scaler
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
# 5. Safe checkpoint loading
# ============================================================
def safe_torch_load(
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
# 6. Backbone
# ============================================================
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


# ============================================================
# 7. Soft hierarchical model
# ============================================================
class SoftHierarchicalClassifier(
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

        # Must remain compatible with the previous binary checkpoint.
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

        binary_logits = self.binary_head(
            feature
        )

        abnormal_logits = self.abnormal_head(
            feature
        )

        four_log_probabilities = (
            build_hierarchical_log_probabilities(
                binary_logits,
                abnormal_logits,
            )
        )

        return {
            "feature": feature,

            "binary_logits": (
                binary_logits
            ),

            "abnormal_logits": (
                abnormal_logits
            ),

            "four_log_probs": (
                four_log_probabilities
            ),
        }


# ============================================================
# 8. Hierarchical probability combination
#
# log P(Normal)
#
# log P(Abnormal subtype)
#     = log P(Abnormal)
#       + log P(subtype | Abnormal)
# ============================================================
def build_hierarchical_log_probabilities(
    binary_logits,
    abnormal_logits,
):
    binary_log_probs = F.log_softmax(
        binary_logits,
        dim=1,
    )

    abnormal_log_probs = F.log_softmax(
        abnormal_logits,
        dim=1,
    )

    normal_log_probability = (
        binary_log_probs[
            :,
            0:1,
        ]
    )

    abnormal_gate_log_probability = (
        binary_log_probs[
            :,
            1:2,
        ]
    )

    conditional_abnormal_log_probabilities = (
        abnormal_gate_log_probability
        + abnormal_log_probs
    )

    four_log_probabilities = torch.cat(
        [
            normal_log_probability,
            conditional_abnormal_log_probabilities,
        ],
        dim=1,
    )

    return four_log_probabilities


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
                f"{csv_path} 缺少列："
                f"{sorted(missing_columns)}；"
                f"当前列："
                f"{self.df.columns.tolist()}"
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
                "标签必须为 0、1、2、3；"
                f"发现："
                f"{invalid_labels.tolist()}"
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
            f"{csv_path}"
        )

    def __len__(
        self,
    ):
        return len(
            self.df
        )

    def resolve_token_path(
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
            self.resolve_token_path(
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
# 11. DataLoader
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
# 12. Abnormal class weights
# ============================================================
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


# ============================================================
# 13. Binary metrics
# ============================================================
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

    score = (
        sp + se
    ) / 2.0

    return {
        f"{prefix}_SP": float(
            sp
        ),

        f"{prefix}_SE": float(
            se
        ),

        f"{prefix}_SCORE": float(
            score
        ),

        f"{prefix}_CM": cm,
    }


# ============================================================
# 14. Strict four-class metrics
# ============================================================
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

        "MACRO_F1": float(
            macro_f1
        ),

        "RECALL": class_recall,

        "PRED_COUNTS": (
            predicted_counts
        ),

        "FOUR_CM": cm,
    }


# ============================================================
# 15. Joint loss
# ============================================================
def calculate_joint_loss(
    outputs,
    labels,
    abnormal_class_weights,
    cfg,
):
    binary_logits = outputs[
        "binary_logits"
    ]

    abnormal_logits = outputs[
        "abnormal_logits"
    ]

    four_log_probs = outputs[
        "four_log_probs"
    ]

    binary_target = (
        labels > 0
    ).long()

    abnormal_mask = (
        labels > 0
    )

    # --------------------------------------------------------
    # Main four-class hierarchical loss
    # --------------------------------------------------------
    four_loss = F.nll_loss(
        four_log_probs,
        labels,
    )

    # --------------------------------------------------------
    # Auxiliary binary loss
    # --------------------------------------------------------
    binary_loss = F.cross_entropy(
        binary_logits,
        binary_target,
    )

    # --------------------------------------------------------
    # Auxiliary abnormal subtype loss
    # --------------------------------------------------------
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

            weight=(
                abnormal_class_weights
            ),
        )

    else:
        abnormal_loss = torch.zeros(
            (),
            dtype=four_loss.dtype,
            device=four_loss.device,
        )

    total_loss = (
        float(
            cfg[
                "FOUR_LOSS_WEIGHT"
            ]
        )
        * four_loss

        +

        float(
            cfg[
                "BINARY_LOSS_WEIGHT"
            ]
        )
        * binary_loss

        +

        float(
            cfg[
                "ABNORMAL_LOSS_WEIGHT"
            ]
        )
        * abnormal_loss
    )

    return {
        "total": total_loss,
        "four": four_loss,
        "binary": binary_loss,
        "abnormal": abnormal_loss,
    }


# ============================================================
# 16. Train one epoch
# ============================================================
def train_one_epoch(
    loader,
    model,
    optimizer,
    device,
    scaler,
    use_amp,
    abnormal_class_weights,
    cfg,
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulation_steps = int(
        cfg["ACCUM_STEPS"]
    )

    number_of_batches = len(
        loader
    )

    total_loss_sum = 0.0
    four_loss_sum = 0.0
    binary_loss_sum = 0.0
    abnormal_loss_sum = 0.0

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

        total_loss_sum += float(
            losses["total"]
            .detach()
            .item()
        )

        four_loss_sum += float(
            losses["four"]
            .detach()
            .item()
        )

        binary_loss_sum += float(
            losses["binary"]
            .detach()
            .item()
        )

        abnormal_loss_sum += float(
            losses["abnormal"]
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
                    cfg[
                        "GRAD_CLIP"
                    ]
                ),
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

    divisor = max(
        number_of_batches,
        1,
    )

    return {
        "TOTAL_LOSS": (
            total_loss_sum
            / divisor
        ),

        "FOUR_LOSS": (
            four_loss_sum
            / divisor
        ),

        "BINARY_LOSS": (
            binary_loss_sum
            / divisor
        ),

        "ABNORMAL_LOSS": (
            abnormal_loss_sum
            / divisor
        ),

        "OPTIMIZER_STEPS": (
            optimizer_steps
        ),
    }


# ============================================================
# 17. Evaluation
# ============================================================
@torch.no_grad()
def evaluate(
    loader,
    model,
    device,
    abnormal_class_weights,
    cfg,
):
    model.eval()

    all_true = []
    all_four_pred = []
    all_gate_binary_pred = []

    total_loss_sum = 0.0
    four_loss_sum = 0.0
    binary_loss_sum = 0.0
    abnormal_loss_sum = 0.0

    number_of_batches = len(
        loader
    )

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
            abnormal_class_weights,
            cfg,
        )

        four_prediction = torch.argmax(
            outputs[
                "four_log_probs"
            ],
            dim=1,
        )

        gate_binary_prediction = torch.argmax(
            outputs[
                "binary_logits"
            ],
            dim=1,
        )

        all_true.append(
            y.detach().cpu()
        )

        all_four_pred.append(
            four_prediction
            .detach()
            .cpu()
        )

        all_gate_binary_pred.append(
            gate_binary_prediction
            .detach()
            .cpu()
        )

        total_loss_sum += float(
            losses["total"].item()
        )

        four_loss_sum += float(
            losses["four"].item()
        )

        binary_loss_sum += float(
            losses["binary"].item()
        )

        abnormal_loss_sum += float(
            losses["abnormal"].item()
        )

    y_true = torch.cat(
        all_true
    ).numpy()

    y_pred_four = torch.cat(
        all_four_pred
    ).numpy()

    y_pred_gate_binary = torch.cat(
        all_gate_binary_pred
    ).numpy()

    # Binary prediction derived from final four-class prediction.
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
        number_of_batches,
        1,
    )

    metrics.update(
        {
            "TOTAL_LOSS": (
                total_loss_sum
                / divisor
            ),

            "FOUR_LOSS": (
                four_loss_sum
                / divisor
            ),

            "BINARY_LOSS": (
                binary_loss_sum
                / divisor
            ),

            "ABNORMAL_LOSS": (
                abnormal_loss_sum
                / divisor
            ),
        }
    )

    return metrics


# ============================================================
# 18. Shape test
# ============================================================
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

    print(
        "[Shape] input:",
        tuple(
            x.shape
        ),
    )

    print(
        "[Shape] feature:",
        tuple(
            outputs[
                "feature"
            ].shape
        ),
    )

    print(
        "[Shape] binary logits:",
        tuple(
            outputs[
                "binary_logits"
            ].shape
        ),
    )

    print(
        "[Shape] abnormal logits:",
        tuple(
            outputs[
                "abnormal_logits"
            ].shape
        ),
    )

    print(
        "[Shape] four log probs:",
        tuple(
            outputs[
                "four_log_probs"
            ].shape
        ),
    )

    if tuple(
        outputs[
            "feature"
        ].shape
    ) != (
        1,
        int(
            cfg["D_MODEL"]
        ),
    ):
        raise RuntimeError(
            "Backbone output shape error."
        )

    if tuple(
        outputs[
            "binary_logits"
        ].shape
    ) != (
        1,
        2,
    ):
        raise RuntimeError(
            "Binary head shape error."
        )

    if tuple(
        outputs[
            "abnormal_logits"
        ].shape
    ) != (
        1,
        3,
    ):
        raise RuntimeError(
            "Abnormal head shape error."
        )

    if tuple(
        outputs[
            "four_log_probs"
        ].shape
    ) != (
        1,
        4,
    ):
        raise RuntimeError(
            "Four-class probability shape error."
        )

    probability_sum = (
        outputs[
            "four_log_probs"
        ]
        .exp()
        .sum(
            dim=1
        )
    )

    print(
        "[Shape] probability sum:",
        probability_sum
        .detach()
        .cpu()
        .tolist(),
    )

    print(
        "[Shape] Passed."
    )


# ============================================================
# 19. Serializable metrics
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
# 20. Save checkpoint
# ============================================================
def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    metrics,
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

            "final_binary_score": float(
                metrics[
                    "FINAL_BINARY_SCORE"
                ]
            ),

            "gate_binary_score": float(
                metrics[
                    "GATE_BINARY_SCORE"
                ]
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


# ============================================================
# 21. Print final results
# ============================================================
def print_final_results(
    metrics,
):
    print()

    print(
        "=" * 80
    )

    print(
        "[FINAL SOFT HIERARCHICAL TEST]"
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
        "Final-prediction Binary Metrics"
    )

    print(
        f"Binary Score: "
        f"{metrics['FINAL_BINARY_SCORE']:.4f}"
    )

    print(
        f"Binary SP: "
        f"{metrics['FINAL_BINARY_SP']:.4f}"
    )

    print(
        f"Binary SE: "
        f"{metrics['FINAL_BINARY_SE']:.4f}"
    )

    print()

    print(
        "Binary-gate Metrics"
    )

    print(
        f"Gate Score: "
        f"{metrics['GATE_BINARY_SCORE']:.4f}"
    )

    print(
        f"Gate SP: "
        f"{metrics['GATE_BINARY_SP']:.4f}"
    )

    print(
        f"Gate SE: "
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
        "Binary gate confusion matrix:"
    )

    print(
        metrics[
            "GATE_BINARY_CM"
        ]
    )


# ============================================================
# 22. Main
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
        bool(
            cfg["REQUIRE_MAMBA"]
        )
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm 导入失败。"
        )

    # ========================================================
    # Paths
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
            "[INFO] Validation file:",
            selection_csv,
        )

    else:
        selection_csv = (
            test_csv
        )

        print(
            "[WARNING] val_index.csv 不存在。"
        )

        print(
            "[WARNING] 当前暂时使用 test_index.csv "
            "进行 checkpoint 选择。"
        )

        print(
            "[WARNING] 正式实验应建立独立验证集，"
            "否则会产生测试集泄漏。"
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
        / "best_soft_hierarchical.pth"
    )

    last_checkpoint_path = (
        save_dir
        / "last_soft_hierarchical.pth"
    )

    # ========================================================
    # Datasets
    # ========================================================
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

    # ========================================================
    # DataLoaders
    # ========================================================
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

    # ========================================================
    # Model
    # ========================================================
    model = SoftHierarchicalClassifier(
        cfg
    ).to(
        device
    )

    # ========================================================
    # Optional warm start from previous binary checkpoint
    # ========================================================
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
        and binary_checkpoint_path.exists()
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
            "[INIT] Loaded previous binary checkpoint:"
        )

        print(
            binary_checkpoint_path
        )

        print(
            f"[INIT] Previous Binary Score="
            f"{binary_checkpoint.get('binary_score', float('nan')):.4f}"
        )

    else:
        print(
            "[INIT] Training from random initialization."
        )

    # All parts are trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    # ========================================================
    # Shape test
    # ========================================================
    run_shape_test(
        train_loader,
        model,
        device,
        cfg,
    )

    # ========================================================
    # Abnormal weights
    # ========================================================
    abnormal_class_weights = (
        build_abnormal_class_weights(
            train_dataset.class_counts,

            cfg[
                "ABNORMAL_WEIGHT_POWER"
            ],
        )
        .to(device)
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

                "lr": float(
                    cfg[
                        "BACKBONE_LR"
                    ]
                ),
            },

            {
                "params": (
                    model.binary_head
                    .parameters()
                ),

                "lr": float(
                    cfg[
                        "BINARY_HEAD_LR"
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
                cfg["EPOCHS"]
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
        and device.type == "cuda"
    )

    scaler = make_scaler(
        use_amp
    )

    # ========================================================
    # Training
    # ========================================================
    best_four_score = -1.0
    best_macro_f1 = -1.0
    best_epoch = -1

    bad_epochs = 0

    print()

    print(
        "=" * 90
    )

    print(
        "SOFT HIERARCHICAL JOINT TRAINING"
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

        train_metrics = train_one_epoch(
            train_loader,
            model,
            optimizer,
            device,
            scaler,
            use_amp,
            abnormal_class_weights,
            cfg,
        )

        validation_metrics = evaluate(
            selection_loader,
            model,
            device,
            abnormal_class_weights,
            cfg,
        )

        current_backbone_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        current_binary_head_lr = (
            optimizer
            .param_groups[1]["lr"]
        )

        current_abnormal_head_lr = (
            optimizer
            .param_groups[2]["lr"]
        )

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

        current_macro_f1 = float(
            validation_metrics[
                "MACRO_F1"
            ]
        )

        improved = (
            current_four_score
            > best_four_score
            + 1e-9
        )

        if (
            not improved
            and abs(
                current_four_score
                - best_four_score
            )
            <= 1e-9
            and current_macro_f1
            > best_macro_f1
        ):
            improved = True

        if improved:
            best_four_score = (
                current_four_score
            )

            best_macro_f1 = (
                current_macro_f1
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
                model,
                optimizer,
                scheduler,
                validation_metrics,
                abnormal_class_weights,
                cfg,
            )

        else:
            bad_epochs += 1

            marker = "-"

        save_checkpoint(
            last_checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            validation_metrics,
            abnormal_class_weights,
            cfg,
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[{marker}] "
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['EPOCHS']} | "

            f"Train "
            f"{train_metrics['TOTAL_LOSS']:.4f} | "

            f"FourLoss "
            f"{train_metrics['FOUR_LOSS']:.4f} | "

            f"BinLoss "
            f"{train_metrics['BINARY_LOSS']:.4f} | "

            f"AbnLoss "
            f"{train_metrics['ABNORMAL_LOSS']:.4f} | "

            f"4-Score "
            f"{validation_metrics['FOUR_SCORE']:.4f} | "

            f"4-SP "
            f"{validation_metrics['FOUR_SP']:.4f} | "

            f"4-SE "
            f"{validation_metrics['FOUR_SE']:.4f} | "

            f"Final-Bin "
            f"{validation_metrics['FINAL_BINARY_SCORE']:.4f} | "

            f"Gate-Bin "
            f"{validation_metrics['GATE_BINARY_SCORE']:.4f} | "

            f"F1 "
            f"{validation_metrics['MACRO_F1']:.4f} | "

            f"LR "
            f"{current_backbone_lr:.8f}/"
            f"{current_binary_head_lr:.8f}/"
            f"{current_abnormal_head_lr:.8f} | "

            f"{elapsed:.1f}s"
        )

        print(
            "    Recall[0,1,2,3]="
            f"{np.round(validation_metrics['RECALL'], 3).tolist()} | "

            f"PredCount="
            f"{validation_metrics['PRED_COUNTS'].tolist()}"
        )

        if (
            bad_epochs
            >= int(
                cfg[
                    "PATIENCE"
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
    # Load best model
    # ========================================================
    best_checkpoint = safe_torch_load(
        best_checkpoint_path,
        device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state"
        ]
    )

    print()

    print(
        f"[Training Completed] "
        f"Best 4-Class Score="
        f"{best_checkpoint['four_score']:.4f}, "
        f"Epoch="
        f"{best_checkpoint['epoch']}"
    )

    # ========================================================
    # Final test
    # ========================================================
    final_metrics = evaluate(
        test_loader,
        model,
        device,
        abnormal_class_weights,
        cfg,
    )

    print_final_results(
        final_metrics
    )

    print()

    print(
        "Best checkpoint:",
        best_checkpoint_path,
    )

    print(
        "Last checkpoint:",
        last_checkpoint_path,
    )


if __name__ == "__main__":
    main()