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
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. 项目路径与模型导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import (
    HAS_MAMBA,
    DTFHybridModel,
)


# ============================================================
# 2. 配置
# ============================================================
CONFIG = {
    # 数据目录
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_fbank"
    ),

    # 模型与训练记录保存目录
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_dtf_stem_hybrid"
    ),

    # 官方协议：完整训练集训练，最后测试一次
    "EPOCHS": 50,

    # DataLoader
    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 4,

    # 环境
    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # Fbank
    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

    # DTF Stem
    "STEM_DIM": 64,

    # Time-Mamba + Frequency-Attention
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

    # 不同模块分别设置学习率
    "FRONTEND_LR": 3e-4,
    "ENCODER_LR": 1e-4,
    "CLASSIFIER_LR": 3e-4,

    "MIN_FRONTEND_LR": 3e-6,
    "MIN_ENCODER_LR": 1e-6,
    "MIN_CLASSIFIER_LR": 3e-6,

    "WARMUP_EPOCHS": 3,
    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # 四分类损失
    # 第一阶段先以四分类为主，避免辅助损失干扰 DTF 消融
    "FOUR_LOSS_WEIGHT": 1.0,
    "BINARY_LOSS_WEIGHT": 0.0,
    "SUBTYPE_LOSS_WEIGHT": 0.0,

    "LABEL_SMOOTHING": 0.0,

    # 类别权重
    "FOUR_WEIGHT_POWER": 0.50,
    "FOUR_WEIGHT_MAX": 2.20,

    "SUBTYPE_WEIGHT_POWER": 0.50,
    "SUBTYPE_WEIGHT_MAX": 2.00,

    # SpecAugment
    "USE_SPECAUGMENT": True,
    "TIME_MASK_MAX": 160,
    "FREQ_MASK_MAX": 48,

    # 每多少个 batch 打印一次
    "PRINT_INTERVAL": 50,
}


# ============================================================
# 3. 随机种子
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 输入尺寸固定，开启 benchmark 可以提高卷积速度
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


# ============================================================
# 4. AMP
# ============================================================
def make_scaler(enabled):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


# ============================================================
# 5. Warmup + Cosine 学习率
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
            + 0.80
            * epoch
            / max(warmup_epochs, 1)
        )

        current_lrs = [
            base_lr * scale
            for base_lr in base_lrs
        ]

    else:
        cosine_total = max(
            total_epochs - warmup_epochs,
            1,
        )

        cosine_step = min(
            epoch - warmup_epochs,
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
            + (base_lr - min_lr)
            * cosine_ratio
            for base_lr, min_lr
            in zip(base_lrs, min_lrs)
        ]

    for group, current_lr in zip(
        optimizer.param_groups,
        current_lrs,
    ):
        group["lr"] = float(current_lr)

    return current_lrs


# ============================================================
# 6. SpecAugment
# ============================================================
def apply_specaugment(
    fbank,
    time_mask_max,
    freq_mask_max,
):
    """
    输入：
        [T, F]

    使用当前频谱均值作为遮挡值。
    """

    x = fbank.clone()

    time_frames = x.shape[0]
    freq_bins = x.shape[1]

    mask_value = x.mean()

    # --------------------------------------------------------
    # 时间遮挡
    # --------------------------------------------------------
    if time_mask_max > 0:
        max_width = min(
            time_mask_max,
            time_frames,
        )

        width = random.randint(
            0,
            max_width,
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
    # 频率遮挡
    # --------------------------------------------------------
    if freq_mask_max > 0:
        max_width = min(
            freq_mask_max,
            freq_bins,
        )

        width = random.randint(
            0,
            max_width,
        )

        if width > 0:
            start = random.randint(
                0,
                freq_bins - width,
            )

            x[
                :,
                start:start + width
            ] = mask_value

    return x


# ============================================================
# 7. Fbank Dataset
# ============================================================
class FbankDataset(Dataset):
    """
    从 CSV 的 fbank_path 读取：

        [798, 128]

    返回：

        x: [1, 798, 128]
        y: 标量标签
    """

    def __init__(
        self,
        csv_path,
        cfg,
        training=False,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.cfg = cfg
        self.training = training

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV不存在：{self.csv_path}"
            )

        self.df = pd.read_csv(
            self.csv_path
        ).reset_index(drop=True)

        required_columns = {
            "fbank_path",
            "label",
        }

        missing_columns = (
            required_columns
            - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{self.csv_path}缺少列："
                f"{sorted(missing_columns)}"
            )

        self.df["label"] = (
            self.df["label"].astype(int)
        )

        self.labels = self.df[
            "label"
        ].to_numpy(dtype=np.int64)

        invalid_labels = np.unique(
            self.labels[
                (self.labels < 0)
                | (self.labels > 3)
            ]
        )

        if len(invalid_labels) > 0:
            raise ValueError(
                f"发现非法标签："
                f"{invalid_labels.tolist()}"
            )

        self.expected_shape = (
            cfg["FBANK_FRAMES"],
            cfg["FBANK_MELS"],
        )

        self.class_counts = np.bincount(
            self.labels,
            minlength=4,
        )

        print(
            f"[FbankDataset] "
            f"samples={len(self.df)} | "
            f"counts={self.class_counts.tolist()} | "
            f"shape={self.expected_shape} | "
            f"training={self.training} | "
            f"csv={self.csv_path}",
            flush=True,
        )

    def __len__(self):
        return len(self.df)

    def resolve_path(self, raw_path):
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

    def __getitem__(self, index):
        row = self.df.iloc[index]

        fbank_path = self.resolve_path(
            row["fbank_path"]
        )

        fbank = np.load(
            fbank_path,
            allow_pickle=False,
        )

        if tuple(fbank.shape) != self.expected_shape:
            raise ValueError(
                f"Fbank尺寸错误：{fbank_path}\n"
                f"当前={tuple(fbank.shape)}，"
                f"要求={self.expected_shape}"
            )

        if not np.isfinite(fbank).all():
            raise ValueError(
                f"Fbank包含NaN或Inf："
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
                freq_mask_max=self.cfg[
                    "FREQ_MASK_MAX"
                ],
            )

        # [T,F] → [C,T,F]
        x = x.unsqueeze(0)

        y = torch.tensor(
            int(row["label"]),
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
    shuffle,
):
    workers = int(
        cfg["NUM_WORKERS"]
    )

    loader_args = {
        "dataset": dataset,
        "batch_size": cfg["BATCH_SIZE"],
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
        loader_args[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        **loader_args
    )


# ============================================================
# 9. 类别权重
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
        / np.maximum(
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


def build_subtype_weights(
    class_counts,
    cfg,
):
    counts = np.asarray(
        class_counts[1:4],
        dtype=np.float64,
    )

    weights = np.power(
        counts.max()
        / np.maximum(
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
# 10. 损失函数
# ============================================================
def calculate_loss(
    logits,
    labels,
    four_weights,
    subtype_weights,
    cfg,
):
    # --------------------------------------------------------
    # 四分类主损失
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
    # Normal / Abnormal 辅助损失
    # --------------------------------------------------------
    binary_logits = torch.stack(
        [
            logits[:, 0],
            torch.logsumexp(
                logits[:, 1:4],
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
    # Crackle / Wheeze / Both 辅助损失
    # --------------------------------------------------------
    abnormal_mask = labels > 0

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
        cfg["FOUR_LOSS_WEIGHT"]
        * four_loss
        + cfg[
            "BINARY_LOSS_WEIGHT"
        ]
        * binary_loss
        + cfg[
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
# 11. 单轮训练
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

    sums = {
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

    total_batches = len(
        loader
    )

    epoch_start = time.time()

    print(
        f"[TRAIN] "
        f"batches={total_batches} | "
        f"batch={cfg['BATCH_SIZE']} | "
        f"accum={cfg['ACCUM_STEPS']}",
        flush=True,
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
            logits = model(
                x
            )

            losses = calculate_loss(
                logits=logits,
                labels=y,
                four_weights=four_weights,
                subtype_weights=subtype_weights,
                cfg=cfg,
            )

            backward_loss = (
                losses["total"]
                / cfg["ACCUM_STEPS"]
            )

        scaler.scale(
            backward_loss
        ).backward()

        for key in sums:
            sums[key] += float(
                losses[key]
                .detach()
                .item()
            )

        completed = (
            batch_index + 1
        )

        should_step = (
            completed
            % cfg["ACCUM_STEPS"]
            == 0
            or completed
            == total_batches
        )

        if should_step:
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
            completed == 1
            or completed
            % cfg[
                "PRINT_INTERVAL"
            ]
            == 0
            or completed
            == total_batches
        ):
            elapsed = (
                time.time()
                - epoch_start
            )

            average_batch_time = (
                elapsed
                / completed
            )

            eta_seconds = (
                total_batches
                - completed
            ) * average_batch_time

            if device.type == "cuda":
                allocated = (
                    torch.cuda
                    .memory_allocated(
                        device
                    )
                    / 1024 ** 3
                )

                reserved = (
                    torch.cuda
                    .memory_reserved(
                        device
                    )
                    / 1024 ** 3
                )
            else:
                allocated = 0.0
                reserved = 0.0

            print(
                f"  Batch "
                f"{completed:04d}/"
                f"{total_batches} | "
                f"Loss "
                f"{losses['total'].item():.4f} | "
                f"Four "
                f"{losses['four'].item():.4f} | "
                f"ETA "
                f"{eta_seconds / 60:.1f}min | "
                f"GPU "
                f"{allocated:.2f}/"
                f"{reserved:.2f}GB",
                flush=True,
            )

    divisor = max(
        total_batches,
        1,
    )

    return {
        key: value / divisor
        for key, value
        in sums.items()
    }


# ============================================================
# 12. ICBHI 指标
# ============================================================
def calculate_metrics(
    y_true,
    y_pred,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
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
            cm[0, 0]
        )
        / normal_total
    )

    sensitivity = (
        100.0
        * float(
            cm[1, 1]
            + cm[2, 2]
            + cm[3, 3]
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
        labels=[0, 1, 2, 3],
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
        "score": float(
            score
        ),

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

        "four_cm": cm,

        "binary_cm": confusion_matrix(
            binary_true,
            binary_pred,
            labels=[0, 1],
        ),
    }


# ============================================================
# 13. 测试集评估
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

        logits = model(
            x
        )

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
# 14. 模型连接测试
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

    tokens = model.extract_tokens(
        x
    )

    logits = model(
        x
    )

    print(
        "[Shape Test] Fbank:",
        tuple(
            x.shape
        ),
    )

    print(
        "[Shape Test] Tokens:",
        tuple(
            tokens.shape
        ),
    )

    print(
        "[Shape Test] Logits:",
        tuple(
            logits.shape
        ),
    )

    print(
        "[Shape Test] Labels:",
        tuple(
            y.shape
        ),
    )

    print(
        "[Shape Test] DTF alpha:",
        model.get_dtf_alpha(),
    )

    assert tuple(
        x.shape[1:]
    ) == (
        1,
        798,
        128,
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

    print(
        "[PASS] train.py与DTF模型连接成功。",
        flush=True,
    )

    model.train()


# ============================================================
# 15. 输出最终测试结果
# ============================================================
def print_final(
    result,
):
    print()
    print("=" * 80)
    print("FINAL OFFICIAL TEST RESULT")
    print("=" * 80)

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
# 16. 主函数
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
            "mamba_ssm导入失败，"
            "不能进行正式训练。"
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
        "[Protocol] 固定训练轮数，"
        "训练完成后测试一次。"
    )

    print(
        "[Input] 直接读取Fbank，"
        "不读取tokens_path，"
        "不加载任何AST权重。"
    )

    save_dir = Path(
        cfg["SAVE_DIR"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_path = (
        save_dir
        / "last_dtf_stem_model.pth"
    )

    final_path = (
        save_dir
        / "final_dtf_stem_model.pth"
    )

    history_path = (
        save_dir
        / "training_history.csv"
    )

    # ========================================================
    # Dataset
    # ========================================================
    train_set = FbankDataset(
        train_csv,
        cfg,
        training=True,
    )

    test_set = FbankDataset(
        test_csv,
        cfg,
        training=False,
    )

    train_loader = make_loader(
        train_set,
        cfg,
        device,
        shuffle=True,
    )

    test_loader = make_loader(
        test_set,
        cfg,
        device,
        shuffle=False,
    )

    # ========================================================
    # Model
    # ========================================================
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
    ).to(
        device
    )

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

    # ========================================================
    # 类别权重
    # ========================================================
    four_weights = build_four_weights(
        train_set.class_counts,
        cfg,
    ).to(
        device
    )

    subtype_weights = build_subtype_weights(
        train_set.class_counts,
        cfg,
    ).to(
        device
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

    base_lrs = [
        cfg["FRONTEND_LR"],
        cfg["ENCODER_LR"],
        cfg["CLASSIFIER_LR"],
    ]

    min_lrs = [
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
    print("=" * 90)
    print(
        "DTF STEM → TIME-MAMBA "
        "→ FREQUENCY-ATTENTION TRAINING"
    )
    print("=" * 90)

    # ========================================================
    # 固定轮数训练
    # ========================================================
    for epoch in range(
        1,
        cfg["EPOCHS"] + 1,
    ):
        epoch_start = time.time()

        current_lrs = set_epoch_lrs(
            optimizer=optimizer,
            base_lrs=base_lrs,
            min_lrs=min_lrs,
            epoch=epoch,
            total_epochs=cfg[
                "EPOCHS"
            ],
            warmup_epochs=cfg[
                "WARMUP_EPOCHS"
            ],
        )

        train_result = train_one_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            four_weights=four_weights,
            subtype_weights=subtype_weights,
            cfg=cfg,
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        current_alpha = (
            model.get_dtf_alpha()
        )

        history_row = {
            "epoch": epoch,

            "total_loss": train_result[
                "total"
            ],

            "four_loss": train_result[
                "four"
            ],

            "binary_loss": train_result[
                "binary"
            ],

            "subtype_loss": train_result[
                "subtype"
            ],

            "frontend_lr": current_lrs[
                0
            ],

            "encoder_lr": current_lrs[
                1
            ],

            "classifier_lr": current_lrs[
                2
            ],

            "dtf_alpha": current_alpha,

            "seconds": elapsed,
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

                "dtf_alpha": (
                    current_alpha
                ),
            },
            last_path,
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{cfg['EPOCHS']} | "
            f"Train "
            f"{train_result['total']:.4f} | "
            f"Four "
            f"{train_result['four']:.4f} | "
            f"Alpha "
            f"{current_alpha:.4f} | "
            f"LR "
            f"{current_lrs[0]:.8f}/"
            f"{current_lrs[1]:.8f}/"
            f"{current_lrs[2]:.8f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    # ========================================================
    # 训练完成后，测试一次
    # ========================================================
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

            "dtf_alpha": (
                model.get_dtf_alpha()
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

            "test_pred_counts": (
                final_result[
                    "pred_counts"
                ].tolist()
            ),

            "test_four_cm": final_result[
                "four_cm"
            ].tolist(),

            "test_binary_cm": final_result[
                "binary_cm"
            ].tolist(),
        },
        final_path,
    )

    print()
    print(
        "Last checkpoint:",
        last_path,
    )

    print(
        "Final checkpoint:",
        final_path,
    )

    print(
        "Training history:",
        history_path,
    )


if __name__ == "__main__":
    main()