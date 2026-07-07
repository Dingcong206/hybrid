#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    Dataset,
)


# ============================================================
# 1. Project Import
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
    DTFHybridModel,
)


# ============================================================
# 2. Configuration
# ============================================================
CONFIG: Dict[str, object] = {
    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_fbank"
    ),

    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_d4_3_residual_bimamba_seed42"
    ),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    "EPOCHS": 50,

    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 4,

    "SEED": 42,

    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------
    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

    # --------------------------------------------------------
    # Model
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
    # Residual BiMamba
    # --------------------------------------------------------
    "BACKWARD_MAX_SCALE": 0.30,
    "BACKWARD_INIT_LOGIT": -2.0,

    # --------------------------------------------------------
    # Original D4 Loss
    # --------------------------------------------------------
    "FOUR_LOSS_WEIGHT": 1.00,
    "BINARY_LOSS_WEIGHT": 0.25,
    "ABNORMAL_LOSS_WEIGHT": 0.75,

    "ABNORMAL_MANUAL_WEIGHTS": [
        1.00,
        1.10,
        1.35,
    ],

    # --------------------------------------------------------
    # Dynamic Hierarchical Fusion
    # --------------------------------------------------------
    "MIN_HIERARCHICAL_WEIGHT": 0.15,
    "MAX_HIERARCHICAL_WEIGHT": 0.60,

    # --------------------------------------------------------
    # SpecAugment
    # --------------------------------------------------------
    "USE_SPECAUGMENT": True,

    "TIME_MASK_MAX": 80,
    "FREQ_MASK_MAX": 16,

    "NUM_TIME_MASKS": 1,
    "NUM_FREQ_MASKS": 1,

    # --------------------------------------------------------
    # Learning Rate
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

    "PRINT_INTERVAL": 50,
}


# ============================================================
# 3. Random Seed
# ============================================================
def set_seed(
    seed: int,
) -> None:
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
        torch.set_float32_matmul_precision(
            "high"
        )
    except AttributeError:
        pass


# ============================================================
# 4. AMP GradScaler
# ============================================================
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
            enabled=enabled,
        )


# ============================================================
# 5. Learning Rate
# ============================================================
def set_epoch_lrs(
    optimizer,
    base_lrs,
    minimum_lrs,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
):
    if epoch <= warmup_epochs:
        warmup_ratio = (
            0.20
            + 0.80
            * epoch
            / max(
                warmup_epochs,
                1,
            )
        )

        current_lrs = [
            base_lr * warmup_ratio
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
# 6. SpecAugment
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
            "SpecAugment输入必须为[T,F]。"
        )

    x = fbank.clone()

    time_frames = int(
        x.shape[0]
    )

    frequency_bins = int(
        x.shape[1]
    )

    mask_value = x.mean()

    for _ in range(
        max(
            num_time_masks,
            0,
        )
    ):
        if time_mask_max <= 0:
            break

        mask_width = random.randint(
            0,
            min(
                time_mask_max,
                time_frames,
            ),
        )

        if mask_width > 0:
            mask_start = random.randint(
                0,
                time_frames
                - mask_width,
            )

            x[
                mask_start:
                mask_start + mask_width,
                :
            ] = mask_value

    for _ in range(
        max(
            num_frequency_masks,
            0,
        )
    ):
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
                frequency_bins
                - mask_width,
            )

            x[
                :,
                mask_start:
                mask_start + mask_width,
            ] = mask_value

    return x


# ============================================================
# 7. Dataset
# ============================================================
class FbankDataset(Dataset):
    def __init__(
        self,
        csv_path,
        cfg,
        training: bool,
    ) -> None:
        super().__init__()

        self.csv_path = Path(
            csv_path
        )

        self.cfg = cfg
        self.training = training

        self.expected_shape = (
            int(
                cfg["FBANK_FRAMES"]
            ),
            int(
                cfg["FBANK_MELS"]
            ),
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(
                self.csv_path
            )

        self.dataframe = pd.read_csv(
            self.csv_path
        ).reset_index(
            drop=True
        )

        required_columns = {
            "fbank_path",
            "label",
        }

        missing_columns = (
            required_columns
            - set(
                self.dataframe.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"CSV缺少列："
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

        class_counts = np.bincount(
            self.labels,
            minlength=4,
        )

        print(
            f"[FbankDataset] "
            f"samples={len(self.dataframe)} | "
            f"counts={class_counts.tolist()} | "
            f"training={self.training}",
            flush=True,
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.dataframe
        )

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
            self.csv_path.parent
            / fbank_path
        )

        if candidate_path.exists():
            return candidate_path

        raise FileNotFoundError(
            f"找不到Fbank文件："
            f"{raw_path}"
        )

    def __getitem__(
        self,
        index: int,
    ):
        row = self.dataframe.iloc[
            index
        ]

        fbank_path = self.resolve_path(
            row["fbank_path"]
        )

        fbank = np.load(
            fbank_path,
            allow_pickle=False,
        )

        if tuple(
            fbank.shape
        ) != self.expected_shape:
            raise ValueError(
                f"Fbank尺寸错误："
                f"{fbank_path}\n"
                f"当前={tuple(fbank.shape)}，"
                f"要求={self.expected_shape}"
            )

        if not np.isfinite(
            fbank
        ).all():
            raise ValueError(
                f"Fbank包含NaN或Inf："
                f"{fbank_path}"
            )

        x = torch.from_numpy(
            fbank
        ).float()

        if (
            self.training
            and bool(
                self.cfg[
                    "USE_SPECAUGMENT"
                ]
            )
        ):
            x = apply_specaugment(
                fbank=x,

                time_mask_max=int(
                    self.cfg[
                        "TIME_MASK_MAX"
                    ]
                ),

                frequency_mask_max=int(
                    self.cfg[
                        "FREQ_MASK_MAX"
                    ]
                ),

                num_time_masks=int(
                    self.cfg[
                        "NUM_TIME_MASKS"
                    ]
                ),

                num_frequency_masks=int(
                    self.cfg[
                        "NUM_FREQ_MASKS"
                    ]
                ),
            )

        # [T,F] -> [1,T,F]
        x = x.unsqueeze(0)

        y = torch.tensor(
            int(
                row["label"]
            ),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# 8. DataLoader
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

        "persistent_workers": (
            workers > 0
        ),

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
# 9. Model
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
            cfg[
                "BINARY_RESIDUAL_SCALE"
            ]
        ),

        backward_max_scale=float(
            cfg[
                "BACKWARD_MAX_SCALE"
            ]
        ),

        backward_init_logit=float(
            cfg[
                "BACKWARD_INIT_LOGIT"
            ]
        ),
    ).to(device)

    return model


# ============================================================
# 10. Loss Weights
# ============================================================
def build_loss_weights(
    cfg,
    device,
):
    abnormal_weight = torch.tensor(
        cfg[
            "ABNORMAL_MANUAL_WEIGHTS"
        ],
        dtype=torch.float32,
        device=device,
    )

    return {
        "four": None,
        "binary": None,
        "abnormal": abnormal_weight,
    }


# ============================================================
# 11. Multi-task Loss
# ============================================================
def calculate_multitask_loss(
    outputs,
    labels,
    loss_weights,
    cfg,
):
    four_loss = F.cross_entropy(
        outputs[
            "four_logits"
        ],
        labels,
        weight=loss_weights["four"],
    )

    binary_labels = (
        labels > 0
    ).long()

    binary_loss = F.cross_entropy(
        outputs[
            "binary_logits"
        ],
        binary_labels,
        weight=loss_weights["binary"],
    )

    abnormal_mask = (
        labels > 0
    )

    abnormal_count = int(
        abnormal_mask.sum().item()
    )

    if abnormal_count > 0:
        abnormal_labels = (
            labels[
                abnormal_mask
            ]
            - 1
        )

        abnormal_logits = outputs[
            "abnormal_logits"
        ][
            abnormal_mask
        ]

        abnormal_loss = F.cross_entropy(
            abnormal_logits,
            abnormal_labels,
            weight=loss_weights[
                "abnormal"
            ],
        )

    else:
        abnormal_loss = (
            outputs[
                "abnormal_logits"
            ].sum()
            * 0.0
        )

    total_loss = (
        float(
            cfg[
                "FOUR_LOSS_WEIGHT"
            ]
        )
        * four_loss

        + float(
            cfg[
                "BINARY_LOSS_WEIGHT"
            ]
        )
        * binary_loss

        + float(
            cfg[
                "ABNORMAL_LOSS_WEIGHT"
            ]
        )
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
# 12. Train One Epoch
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
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_batches = len(
        loader
    )

    accumulation_steps = int(
        cfg["ACCUM_STEPS"]
    )

    total_samples = 0
    total_loss_sum = 0.0

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    epoch_start_time = time.time()

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

        completed_batches = (
            batch_index + 1
        )

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
            outputs = model(
                x
            )

            loss_result = (
                calculate_multitask_loss(
                    outputs=outputs,
                    labels=y,
                    loss_weights=loss_weights,
                    cfg=cfg,
                )
            )

            backward_loss = (
                loss_result[
                    "total_loss"
                ]
                / current_accumulation_size
            )

        scaler.scale(
            backward_loss
        ).backward()

        batch_size = int(
            y.shape[0]
        )

        total_samples += batch_size

        total_loss_sum += (
            float(
                loss_result[
                    "total_loss"
                ].detach().item()
            )
            * batch_size
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
                    cfg[
                        "GRAD_CLIP"
                    ]
                ),
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        if (
            completed_batches == 1
            or completed_batches
            % int(
                cfg[
                    "PRINT_INTERVAL"
                ]
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

            print(
                f"  Batch "
                f"{completed_batches:04d}/"
                f"{total_batches} | "

                f"Loss "
                f"{loss_result['total_loss'].item():.4f} | "

                f"ETA "
                f"{remaining_seconds / 60:.1f}min",
                flush=True,
            )

    return (
        total_loss_sum
        / max(
            total_samples,
            1,
        )
    )


# ============================================================
# 13. Final Evaluation
# ============================================================
@torch.no_grad()
def evaluate(
    loader,
    model,
    device,
    cfg,
    use_amp: bool,
):
    model.eval()

    all_labels = []
    all_predictions = []

    for x, y in loader:
        x = x.to(
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

            probabilities = (
                model.build_probabilities(
                    outputs=outputs,

                    four_weight=None,

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

        all_labels.append(
            y.cpu()
        )

        all_predictions.append(
            prediction.cpu()
        )

    y_true = torch.cat(
        all_labels
    ).numpy()

    y_pred = torch.cat(
        all_predictions
    ).numpy()

    normal_mask = (
        y_true == 0
    )

    abnormal_mask = (
        y_true > 0
    )

    specificity = (
        100.0
        * float(
            np.sum(
                y_pred[
                    normal_mask
                ] == 0
            )
        )
        / max(
            int(
                np.sum(
                    normal_mask
                )
            ),
            1,
        )
    )

    sensitivity = (
        100.0
        * float(
            np.sum(
                y_pred[
                    abnormal_mask
                ]
                == y_true[
                    abnormal_mask
                ]
            )
        )
        / max(
            int(
                np.sum(
                    abnormal_mask
                )
            ),
            1,
        )
    )

    score = (
        specificity
        + sensitivity
    ) / 2.0

    return {
        "score": float(
            score
        ),

        "sp": float(
            specificity
        ),

        "se": float(
            sensitivity
        ),
    }


# ============================================================
# 14. Shape Test
# ============================================================
@torch.no_grad()
def shape_test(
    loader,
    model,
    device,
):
    model.eval()

    x, _ = next(
        iter(loader)
    )

    x = x[:2].to(
        device
    )

    outputs = model(
        x
    )

    assert tuple(
        outputs[
            "four_logits"
        ].shape
    ) == (
        2,
        4,
    )

    assert tuple(
        outputs[
            "binary_logits"
        ].shape
    ) == (
        2,
        2,
    )

    assert tuple(
        outputs[
            "abnormal_logits"
        ].shape
    ) == (
        2,
        3,
    )

    print(
        "[PASS] D4.3模型连接成功。"
    )


# ============================================================
# 15. Main
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
            str(
                cfg["DEVICE"]
            ) == "cuda"
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

    print(
        "[INFO] HAS_MAMBA:",
        HAS_MAMBA,
    )

    if (
        bool(
            cfg[
                "REQUIRE_MAMBA"
            ]
        )
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm导入失败。"
        )

    root = Path(
        str(
            cfg["ROOT"]
        )
    )

    train_csv = (
        root
        / "train_index.csv"
    )

    test_csv = (
        root
        / "test_index.csv"
    )

    print(
        "[Protocol] 使用完整官方训练集。"
    )

    print(
        "[Protocol] 固定训练50轮。"
    )

    print(
        "[Experiment] D4.3 Residual BiMamba"
    )

    train_dataset = FbankDataset(
        csv_path=train_csv,
        cfg=cfg,
        training=True,
    )

    test_dataset = FbankDataset(
        csv_path=test_csv,
        cfg=cfg,
        training=False,
    )

    train_loader = make_loader(
        dataset=train_dataset,
        cfg=cfg,
        device=device,
        shuffle=True,
    )

    test_loader = make_loader(
        dataset=test_dataset,
        cfg=cfg,
        device=device,
        shuffle=False,
    )

    model = build_model(
        cfg,
        device,
    )

    shape_test(
        train_loader,
        model,
        device,
    )

    loss_weights = build_loss_weights(
        cfg,
        device,
    )

    head_parameters = (
        list(
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
                    cfg[
                        "FRONTEND_LR"
                    ]
                ),
            },

            {
                "params": (
                    model.encoder.parameters()
                ),
                "lr": float(
                    cfg[
                        "ENCODER_LR"
                    ]
                ),
            },

            {
                "params": (
                    head_parameters
                ),
                "lr": float(
                    cfg[
                        "HEAD_LR"
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

    base_lrs = [
        float(
            cfg["FRONTEND_LR"]
        ),
        float(
            cfg["ENCODER_LR"]
        ),
        float(
            cfg["HEAD_LR"]
        ),
    ]

    minimum_lrs = [
        float(
            cfg[
                "MIN_FRONTEND_LR"
            ]
        ),
        float(
            cfg[
                "MIN_ENCODER_LR"
            ]
        ),
        float(
            cfg[
                "MIN_HEAD_LR"
            ]
        ),
    ]

    scaler = make_scaler(
        use_amp
    )

    save_directory = Path(
        str(
            cfg["SAVE_DIR"]
        )
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        save_directory
        / "training_history.csv"
    )

    final_model_path = (
        save_directory
        / "final_model.pth"
    )

    history = []

    print()
    print("=" * 80)
    print(
        "D4.3 RESIDUAL BIDIRECTIONAL MAMBA TRAINING"
    )
    print("=" * 80)

    for epoch in range(
        1,
        int(
            cfg["EPOCHS"]
        )
        + 1,
    ):
        current_lrs = set_epoch_lrs(
            optimizer=optimizer,
            base_lrs=base_lrs,
            minimum_lrs=minimum_lrs,
            epoch=epoch,
            total_epochs=int(
                cfg["EPOCHS"]
            ),
            warmup_epochs=int(
                cfg[
                    "WARMUP_EPOCHS"
                ]
            ),
        )

        epoch_start_time = time.time()

        train_loss = train_one_epoch(
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

        backward_scales = (
            model.get_backward_scales()
        )

        history.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "frontend_lr": (
                    current_lrs[0]
                ),
                "encoder_lr": (
                    current_lrs[1]
                ),
                "head_lr": (
                    current_lrs[2]
                ),
                "dtf_alpha": (
                    model.get_dtf_alpha()
                ),
                "backward_scale": (
                    backward_scales[0]
                ),
                "seconds": elapsed_time,
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['EPOCHS']} | "

            f"Loss "
            f"{train_loss:.4f} | "

            f"DTF "
            f"{model.get_dtf_alpha():.4f} | "

            f"BiScale "
            f"{backward_scales[0]:.4f} | "

            f"{elapsed_time:.1f}s",
            flush=True,
        )

    final_result = evaluate(
        loader=test_loader,
        model=model,
        device=device,
        cfg=cfg,
        use_amp=use_amp,
    )

    print()
    print("=" * 80)
    print(
        "FINAL OFFICIAL TEST RESULT"
    )
    print("=" * 80)

    print(
        f"ICBHI Score: "
        f"{final_result['score']:.4f}"
    )

    print(
        f"Specificity: "
        f"{final_result['sp']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{final_result['se']:.4f}"
    )

    torch.save(
        {
            "epoch": int(
                cfg["EPOCHS"]
            ),

            "model_state": (
                model.state_dict()
            ),

            "config": deepcopy(
                cfg
            ),

            "score": (
                final_result[
                    "score"
                ]
            ),

            "sp": (
                final_result[
                    "sp"
                ]
            ),

            "se": (
                final_result[
                    "se"
                ]
            ),

            "dtf_alpha": (
                model.get_dtf_alpha()
            ),

            "backward_scales": (
                model.get_backward_scales()
            ),
        },
        final_model_path,
    )

    print()
    print(
        "History:",
        history_path,
    )

    print(
        "Model:",
        final_model_path,
    )


if __name__ == "__main__":
    main()