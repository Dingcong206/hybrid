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
# 项目路径
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
# 配置
# ============================================================
CONFIG = {
    # --------------------------------------------------------
    # 数据
    # --------------------------------------------------------
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_fbank"
    ),

    # --------------------------------------------------------
    # 新实验保存目录
    # --------------------------------------------------------
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_d1_b1_no_weight_weak_aug_seed42"
    ),

    # --------------------------------------------------------
    # 官方协议
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
    # Fbank
    # --------------------------------------------------------
    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

    # --------------------------------------------------------
    # DTF前端
    # --------------------------------------------------------
    "STEM_DIM": 64,

    # 当前B1：关闭TF-MBConv
    "TF_MBCONV_DEPTH": 0,

    "TF_EXPAND_RATIO": 2,
    "TF_SE_REDUCTION": 4,
    "MAX_DROP_PATH": 0.05,

    # --------------------------------------------------------
    # Time-Mamba + Frequency-Attention
    # --------------------------------------------------------
    "D_MODEL": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,

    "D_STATE": 16,
    "D_CONV": 4,
    "EXPAND": 2,

    "DROPOUT": 0.15,
    "HEAD_DROPOUT": 0.20,

    # --------------------------------------------------------
    # 学习率
    # --------------------------------------------------------
    "FRONTEND_LR": 3e-4,
    "ENCODER_LR": 1e-4,
    "CLASSIFIER_LR": 3e-4,

    "MIN_FRONTEND_LR": 3e-6,
    "MIN_ENCODER_LR": 1e-6,
    "MIN_CLASSIFIER_LR": 3e-6,

    "WARMUP_EPOCHS": 3,

    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # --------------------------------------------------------
    # 损失
    #
    # 当前实验关闭类别权重
    # --------------------------------------------------------
    "USE_CLASS_WEIGHTS": False,

    "LABEL_SMOOTHING": 0.0,

    # --------------------------------------------------------
    # 弱化SpecAugment
    # --------------------------------------------------------
    "USE_SPECAUGMENT": True,

    "TIME_MASK_MAX": 80,
    "FREQ_MASK_MAX": 16,

    # --------------------------------------------------------
    # 日志
    # --------------------------------------------------------
    "PRINT_INTERVAL": 50,
}


# ============================================================
# 随机种子
# ============================================================
def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保持与前面实验相同
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
# AMP
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
# Warmup + Cosine LR
# ============================================================
def set_epoch_lrs(
    optimizer,
    base_lrs,
    min_lrs,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
):
    if epoch <= warmup_epochs:
        scale = (
            0.20
            + 0.80
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

        cosine_ratio = 0.5 * (
            1.0
            + math.cos(
                math.pi
                * cosine_step
                / cosine_total
            )
        )

        current_lrs = [
            min_lr
            + (
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
        parameter_group["lr"] = float(
            current_lr
        )

    return current_lrs


# ============================================================
# SpecAugment
# ============================================================
def apply_specaugment(
    fbank: torch.Tensor,
    time_mask_max: int,
    frequency_mask_max: int,
) -> torch.Tensor:
    """
    输入：
        [T,F]

    当前弱增强：
        Time最大遮挡80帧
        Frequency最大遮挡16个频带
    """

    x = fbank.clone()

    time_frames = x.shape[0]
    frequency_bins = x.shape[1]

    mask_value = x.mean()

    # --------------------------------------------------------
    # Time Mask
    # --------------------------------------------------------
    if time_mask_max > 0:
        width = random.randint(
            0,
            min(
                time_mask_max,
                time_frames,
            ),
        )

        if width > 0:
            start = random.randint(
                0,
                time_frames - width,
            )

            x[
                start:start + width,
                :
            ] = mask_value

    # --------------------------------------------------------
    # Frequency Mask
    # --------------------------------------------------------
    if frequency_mask_max > 0:
        width = random.randint(
            0,
            min(
                frequency_mask_max,
                frequency_bins,
            ),
        )

        if width > 0:
            start = random.randint(
                0,
                frequency_bins - width,
            )

            x[
                :,
                start:start + width
            ] = mask_value

    return x


# ============================================================
# Dataset
# ============================================================
class FbankDataset(Dataset):
    def __init__(
        self,
        csv_path,
        cfg,
        training: bool = False,
    ) -> None:
        super().__init__()

        self.csv_path = Path(
            csv_path
        )

        self.cfg = cfg
        self.training = training

        self.expected_shape = (
            cfg["FBANK_FRAMES"],
            cfg["FBANK_MELS"],
        )

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV不存在：{self.csv_path}"
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
                f"{self.csv_path}缺少列："
                f"{sorted(missing_columns)}"
            )

        self.dataframe["label"] = (
            self.dataframe[
                "label"
            ].astype(int)
        )

        self.labels = self.dataframe[
            "label"
        ].to_numpy(
            dtype=np.int64
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

        if len(invalid_labels) > 0:
            raise ValueError(
                "发现非法标签："
                f"{invalid_labels.tolist()}"
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
            f"training={self.training} | "
            f"csv={self.csv_path}",
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

        relative_path = (
            self.csv_path.parent
            / fbank_path
        )

        if relative_path.exists():
            return relative_path

        raise FileNotFoundError(
            f"Fbank文件不存在：{raw_path}"
        )

    def __getitem__(
        self,
        index,
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
                f"Fbank尺寸错误：{fbank_path}\n"
                f"当前={tuple(fbank.shape)}，"
                f"要求={self.expected_shape}"
            )

        if not np.isfinite(
            fbank
        ).all():
            raise ValueError(
                "Fbank包含NaN或Inf："
                f"{fbank_path}"
            )

        x = torch.from_numpy(
            fbank
        ).float()

        if (
            self.training
            and self.cfg[
                "USE_SPECAUGMENT"
            ]
        ):
            x = apply_specaugment(
                x,
                time_mask_max=self.cfg[
                    "TIME_MASK_MAX"
                ],
                frequency_mask_max=self.cfg[
                    "FREQ_MASK_MAX"
                ],
            )

        # [T,F] -> [1,T,F]
        x = x.unsqueeze(0)

        y = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# DataLoader
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

        "batch_size": cfg[
            "BATCH_SIZE"
        ],

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
# 类别权重
# ============================================================
def build_class_weights(
    class_counts,
    cfg,
    device,
):
    if not cfg["USE_CLASS_WEIGHTS"]:
        print(
            "[Loss] 不使用类别权重。",
            flush=True,
        )

        return None

    counts = np.asarray(
        class_counts,
        dtype=np.float64,
    )

    weights = (
        counts.sum()
        / (
            len(counts)
            * np.maximum(
                counts,
                1.0,
            )
        )
    )

    weights = (
        weights
        / weights.mean()
    )

    weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print(
        "[Loss] class weights:",
        np.round(
            weights,
            6,
        ).tolist(),
        flush=True,
    )

    return weight_tensor


# ============================================================
# Loss
# ============================================================
def calculate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights,
    cfg,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=cfg[
            "LABEL_SMOOTHING"
        ],
    )


# ============================================================
# 单轮训练
# ============================================================
def train_one_epoch(
    loader,
    model,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    class_weights,
    cfg,
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    total_batches = len(loader)

    epoch_start_time = time.time()

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    print(
        f"[TRAIN] "
        f"batches={total_batches} | "
        f"batch={cfg['BATCH_SIZE']} | "
        f"accum={cfg['ACCUM_STEPS']}",
        flush=True,
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

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(x)

            loss = calculate_loss(
                logits=logits,
                labels=y,
                class_weights=class_weights,
                cfg=cfg,
            )

            backward_loss = (
                loss
                / cfg["ACCUM_STEPS"]
            )

        scaler.scale(
            backward_loss
        ).backward()

        total_loss += float(
            loss.detach().item()
        )

        completed_batches = (
            batch_index + 1
        )

        should_update = (
            completed_batches
            % cfg["ACCUM_STEPS"]
            == 0
            or completed_batches
            == total_batches
        )

        if should_update:
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                cfg["GRAD_CLIP"],
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
            % cfg["PRINT_INTERVAL"]
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
                    torch.cuda
                    .memory_allocated(
                        device
                    )
                    / 1024 ** 3
                )

                reserved_memory = (
                    torch.cuda
                    .memory_reserved(
                        device
                    )
                    / 1024 ** 3
                )
            else:
                allocated_memory = 0.0
                reserved_memory = 0.0

            print(
                f"  Batch "
                f"{completed_batches:04d}/"
                f"{total_batches} | "
                f"Loss {loss.item():.4f} | "
                f"ETA "
                f"{remaining_seconds / 60:.1f}min | "
                f"GPU "
                f"{allocated_memory:.2f}/"
                f"{reserved_memory:.2f}GB",
                flush=True,
            )

    return (
        total_loss
        / max(
            total_batches,
            1,
        )
    )


# ============================================================
# ICBHI指标
# ============================================================
def calculate_metrics(
    y_true,
    y_pred,
):
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

    normal_total = max(
        int(
            confusion[0].sum()
        ),
        1,
    )

    abnormal_total = max(
        int(
            confusion[1:].sum()
        ),
        1,
    )

    specificity = (
        100.0
        * float(
            confusion[0, 0]
        )
        / normal_total
    )

    sensitivity = (
        100.0
        * float(
            confusion[1, 1]
            + confusion[2, 2]
            + confusion[3, 3]
        )
        / abnormal_total
    )

    score = (
        specificity
        + sensitivity
    ) / 2.0

    recalls = recall_score(
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

    binary_true = (
        y_true > 0
    ).astype(
        np.int64
    )

    binary_pred = (
        y_pred > 0
    ).astype(
        np.int64
    )

    return {
        "score": float(score),

        "sp": float(
            specificity
        ),

        "se": float(
            sensitivity
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            ) * 100.0
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "recalls": recalls,

        "pred_counts": np.bincount(
            y_pred,
            minlength=4,
        ),

        "four_cm": confusion,

        "binary_cm": confusion_matrix(
            binary_true,
            binary_pred,
            labels=[
                0,
                1,
            ],
        ),
    }


# ============================================================
# 测试
# ============================================================
@torch.no_grad()
def evaluate(
    loader,
    model,
    device,
):
    model.eval()

    all_labels = []
    all_predictions = []

    for x, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        logits = model(x)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.append(
            y.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

    y_true = torch.cat(
        all_labels
    ).numpy()

    y_pred = torch.cat(
        all_predictions
    ).numpy()

    return calculate_metrics(
        y_true,
        y_pred,
    )


# ============================================================
# Shape Test
# ============================================================
@torch.no_grad()
def shape_test(
    loader,
    model,
    device,
):
    model.eval()

    x, y = next(
        iter(loader)
    )

    x = x[:2].to(
        device
    )

    y = y[:2].to(
        device
    )

    (
        tokens,
        stem_map,
        block_map,
        patch_map,
    ) = model.frontend(
        x,
        return_maps=True,
    )

    logits = model(x)

    alpha_values = (
        model.get_all_alphas()
    )

    print(
        "[Shape Test] Fbank:",
        tuple(x.shape),
    )

    print(
        "[Shape Test] DTF Stem Map:",
        tuple(stem_map.shape),
    )

    print(
        "[Shape Test] Block Map:",
        tuple(block_map.shape),
    )

    print(
        "[Shape Test] Patch Map:",
        tuple(patch_map.shape),
    )

    print(
        "[Shape Test] Tokens:",
        tuple(tokens.shape),
    )

    print(
        "[Shape Test] Logits:",
        tuple(logits.shape),
    )

    print(
        "[Shape Test] Labels:",
        tuple(y.shape),
    )

    print(
        "[Shape Test] Alphas:",
        alpha_values,
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
        block_map.shape[1:]
    ) == (
        64,
        399,
        64,
    )

    assert tuple(
        patch_map.shape[1:]
    ) == (
        256,
        79,
        12,
    )

    assert tuple(
        tokens.shape[1:]
    ) == (
        948,
        256,
    )

    assert tuple(
        logits.shape[1:]
    ) == (
        4,
    )

    assert set(
        alpha_values.keys()
    ) == {
        "stem_alpha",
    }

    print(
        "[PASS] B1模型连接成功，"
        "当前没有启用TF-MBConv。",
        flush=True,
    )

    model.train()


# ============================================================
# 输出结果
# ============================================================
def print_final(
    result,
) -> None:
    print()

    print(
        "=" * 80
    )

    print(
        "FINAL OFFICIAL TEST RESULT"
    )

    print(
        "=" * 80
    )

    print(
        f"ICBHI Score: "
        f"{result['score']:.4f}"
    )

    print(
        f"Specificity: "
        f"{result['sp']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{result['se']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{result['macro_f1']:.4f}"
    )

    print(
        "Recall "
        "[Normal, Crackle, Wheeze, Both]:",
        np.round(
            result["recalls"],
            4,
        ).tolist(),
    )

    print(
        "PredCount:",
        result[
            "pred_counts"
        ].tolist(),
    )

    print()
    print(
        "Four-class confusion matrix:"
    )

    print(
        result[
            "four_cm"
        ]
    )

    print()
    print(
        "Binary confusion matrix:"
    )

    print(
        result[
            "binary_cm"
        ]
    )


# ============================================================
# 主函数
# ============================================================
def main() -> None:
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
                0
            ),
        )

    print(
        "[INFO] HAS_MAMBA:",
        HAS_MAMBA,
    )

    if (
        cfg["REQUIRE_MAMBA"]
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm导入失败。"
        )

    root = Path(
        cfg["ROOT"]
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

    print(
        "[Protocol] 使用完整官方训练集。"
    )

    print(
        "[Protocol] 不划分验证集。"
    )

    print(
        "[Protocol] 固定训练50轮，"
        "最后测试一次。"
    )

    print(
        "[Input] 直接读取Fbank，"
        "不使用AST Token。"
    )

    print(
        "[Experiment] B1 DTF Stem"
        " + No Class Weight"
        " + Weak SpecAugment(80/16)"
    )

    save_dir = Path(
        cfg["SAVE_DIR"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_checkpoint_path = (
        save_dir
        / "last_model.pth"
    )

    final_checkpoint_path = (
        save_dir
        / "final_model.pth"
    )

    history_path = (
        save_dir
        / "training_history.csv"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset = FbankDataset(
        train_csv,
        cfg,
        training=True,
    )

    test_dataset = FbankDataset(
        test_csv,
        cfg,
        training=False,
    )

    train_loader = make_loader(
        train_dataset,
        cfg,
        device,
        shuffle=True,
    )

    test_loader = make_loader(
        test_dataset,
        cfg,
        device,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = DTFHybridModel(
        num_classes=4,

        stem_dim=cfg[
            "STEM_DIM"
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

        tf_mbconv_depth=cfg[
            "TF_MBCONV_DEPTH"
        ],

        tf_expand_ratio=cfg[
            "TF_EXPAND_RATIO"
        ],

        tf_se_reduction=cfg[
            "TF_SE_REDUCTION"
        ],

        max_drop_path=cfg[
            "MAX_DROP_PATH"
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

        head_dropout=cfg[
            "HEAD_DROPOUT"
        ],

        d_state=cfg[
            "D_STATE"
        ],

        d_conv=cfg[
            "D_CONV"
        ],

        expand=cfg[
            "EXPAND"
        ],
    ).to(device)

    shape_test(
        train_loader,
        model,
        device,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
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

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    class_weights = build_class_weights(
        train_dataset.class_counts,
        cfg,
        device,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model
                    .frontend
                    .parameters()
                ),

                "lr": cfg[
                    "FRONTEND_LR"
                ],
            },

            {
                "params": (
                    model
                    .encoder
                    .parameters()
                ),

                "lr": cfg[
                    "ENCODER_LR"
                ],
            },

            {
                "params": (
                    model
                    .classifier
                    .parameters()
                ),

                "lr": cfg[
                    "CLASSIFIER_LR"
                ],
            },
        ],

        weight_decay=cfg[
            "WEIGHT_DECAY"
        ],
    )

    base_learning_rates = [
        cfg["FRONTEND_LR"],
        cfg["ENCODER_LR"],
        cfg["CLASSIFIER_LR"],
    ]

    minimum_learning_rates = [
        cfg["MIN_FRONTEND_LR"],
        cfg["MIN_ENCODER_LR"],
        cfg["MIN_CLASSIFIER_LR"],
    ]

    use_amp = bool(
        cfg["AMP"]
        and device.type == "cuda"
    )

    scaler = make_scaler(
        use_amp
    )

    history = []

    print()
    print(
        "=" * 90
    )

    print(
        "B1: DTF STEM"
        " -> TIME-MAMBA"
        " -> FREQUENCY-ATTENTION"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # 固定50轮训练
    # --------------------------------------------------------
    for epoch in range(
        1,
        cfg["EPOCHS"] + 1,
    ):
        epoch_start_time = time.time()

        current_learning_rates = set_epoch_lrs(
            optimizer=optimizer,

            base_lrs=base_learning_rates,

            min_lrs=minimum_learning_rates,

            epoch=epoch,

            total_epochs=cfg[
                "EPOCHS"
            ],

            warmup_epochs=cfg[
                "WARMUP_EPOCHS"
            ],
        )

        train_loss = train_one_epoch(
            loader=train_loader,

            model=model,

            optimizer=optimizer,

            device=device,

            scaler=scaler,

            use_amp=use_amp,

            class_weights=class_weights,

            cfg=cfg,
        )

        elapsed_time = (
            time.time()
            - epoch_start_time
        )

        alpha_values = (
            model.get_all_alphas()
        )

        stem_alpha = alpha_values[
            "stem_alpha"
        ]

        history_row = {
            "epoch": epoch,

            "train_loss": (
                train_loss
            ),

            "frontend_lr": (
                current_learning_rates[0]
            ),

            "encoder_lr": (
                current_learning_rates[1]
            ),

            "classifier_lr": (
                current_learning_rates[2]
            ),

            "stem_alpha": (
                stem_alpha
            ),

            "seconds": elapsed_time,
        }

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        torch.save(
            {
                "epoch": epoch,

                "model_state": (
                    model.state_dict()
                ),

                "optimizer_state": (
                    optimizer.state_dict()
                ),

                "config": deepcopy(
                    cfg
                ),

                "alpha_values": (
                    alpha_values
                ),
            },
            last_checkpoint_path,
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['EPOCHS']} | "
            f"Train "
            f"{train_loss:.4f} | "
            f"StemAlpha "
            f"{stem_alpha:.4f} | "
            f"LR "
            f"{current_learning_rates[0]:.8f}/"
            f"{current_learning_rates[1]:.8f}/"
            f"{current_learning_rates[2]:.8f} | "
            f"{elapsed_time:.1f}s",
            flush=True,
        )

    # --------------------------------------------------------
    # 官方测试集：训练结束后只测试一次
    # --------------------------------------------------------
    final_result = evaluate(
        test_loader,
        model,
        device,
    )

    print_final(
        final_result
    )

    torch.save(
        {
            "epoch": cfg[
                "EPOCHS"
            ],

            "model_state": (
                model.state_dict()
            ),

            "config": deepcopy(
                cfg
            ),

            "alpha_values": (
                model.get_all_alphas()
            ),

            "test_score": final_result[
                "score"
            ],

            "test_sp": final_result[
                "sp"
            ],

            "test_se": final_result[
                "se"
            ],

            "test_accuracy": final_result[
                "accuracy"
            ],

            "test_macro_f1": final_result[
                "macro_f1"
            ],

            "test_recalls": final_result[
                "recalls"
            ].tolist(),

            "test_pred_counts": final_result[
                "pred_counts"
            ].tolist(),

            "test_four_cm": final_result[
                "four_cm"
            ].tolist(),

            "test_binary_cm": final_result[
                "binary_cm"
            ].tolist(),
        },
        final_checkpoint_path,
    )

    print()

    print(
        "Last checkpoint:",
        last_checkpoint_path,
    )

    print(
        "Final checkpoint:",
        final_checkpoint_path,
    )

    print(
        "Training history:",
        history_path,
    )


if __name__ == "__main__":
    main()