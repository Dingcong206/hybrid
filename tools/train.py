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
    # 直接读取 Fbank
    "ROOT": "/data/dingcong/hybrid/icbhi_official_fbank",

    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_pure_fbank_baseline"
    ),

    # DataLoader
    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 4,
    "NUM_WORKERS": 4,

    # 官方协议：不划分验证集，固定轮数训练
    "EPOCHS": 50,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # Fbank 输入
    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

    # Patch Embedding
    #
    # 输入调整为 [B, 1, F, T] = [B,1,128,798]
    # kernel=16, stride=10
    #
    # Frequency:
    # floor((128 - 16) / 10) + 1 = 12
    #
    # Time:
    # floor((798 - 16) / 10) + 1 = 79
    #
    # 总 Patch 数：
    # 12 × 79 = 948
    "PATCH_KERNEL": (16, 16),
    "PATCH_STRIDE": (10, 10),
    "PATCH_DIM": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    # Time-Mamba + Frequency-Attention
    "D_MODEL": 256,
    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,
    "NHEAD": 8,
    "DROPOUT": 0.15,

    # 分类头
    "HEAD_DROPOUT": 0.20,

    # 学习率
    # Patch Embedding 是随机初始化，学习率稍大
    "PATCH_LR": 3e-4,
    "BACKBONE_LR": 1e-4,
    "HEAD_LR": 3e-4,

    "MIN_PATCH_LR": 3e-6,
    "MIN_BACKBONE_LR": 1e-6,
    "MIN_HEAD_LR": 3e-6,

    "WARMUP_EPOCHS": 3,
    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # 联合损失，暂时保持之前的形式
    "FOUR_LOSS_WEIGHT": 1.0,
    "BINARY_LOSS_WEIGHT": 0.05,
    "SUBTYPE_LOSS_WEIGHT": 1.0,
    "LABEL_SMOOTHING": 0.0,

    # 类别权重
    "FOUR_WEIGHT_POWER": 0.50,
    "FOUR_WEIGHT_MAX": 2.20,

    "SUBTYPE_WEIGHT_POWER": 0.50,
    "SUBTYPE_WEIGHT_MAX": 2.00,

    # 先不使用 SpecAugment，保证只测试输入替换
    "USE_SPECAUGMENT": False,
    "TIME_MASK_MAX": 160,
    "FREQ_MASK_MAX": 48,

    # 每隔多少 Batch 输出一次进度
    "PRINT_INTERVAL": 50,
}


# ============================================================
# 2. 导入原来的 Time-Mamba + Frequency-Attention
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import (
    HAS_MAMBA,
    TimeFrequencyEncoder,
)


# ============================================================
# 3. 随机种子
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
def make_scaler(enabled: bool):
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
# 5. 学习率：Warmup + Cosine
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

    for parameter_group, current_lr in zip(
        optimizer.param_groups,
        current_lrs,
    ):
        parameter_group["lr"] = float(current_lr)

    return current_lrs


# ============================================================
# 6. SpecAugment
# ============================================================
def apply_specaugment(
    fbank: torch.Tensor,
    time_mask_max: int,
    freq_mask_max: int,
) -> torch.Tensor:
    """
    输入：
        [T, F]

    使用当前频谱平均值作为遮挡值。
    """
    fbank = fbank.clone()

    time_frames, freq_bins = fbank.shape

    mask_value = fbank.mean()

    # Time Mask
    if time_mask_max > 0:
        time_width = random.randint(
            0,
            min(time_mask_max, time_frames),
        )

        if time_width > 0:
            time_start = random.randint(
                0,
                time_frames - time_width,
            )

            fbank[
                time_start:time_start + time_width,
                :
            ] = mask_value

    # Frequency Mask
    if freq_mask_max > 0:
        freq_width = random.randint(
            0,
            min(freq_mask_max, freq_bins),
        )

        if freq_width > 0:
            freq_start = random.randint(
                0,
                freq_bins - freq_width,
            )

            fbank[
                :,
                freq_start:freq_start + freq_width,
            ] = mask_value

    return fbank


# ============================================================
# 7. Fbank Dataset
# ============================================================
class FbankDataset(Dataset):
    """
    从 CSV 的 fbank_path 读取：

        [798, 128]

    返回：

        [1, 798, 128]
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
                f"CSV 不存在：{self.csv_path}"
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
                f"{self.csv_path} 缺少列："
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
        path = Path(str(raw_path))

        if path.exists():
            return path

        relative_path = (
            self.csv_path.parent / path
        )

        if relative_path.exists():
            return relative_path

        raise FileNotFoundError(
            f"Fbank 文件不存在：{raw_path}"
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
                f"Fbank shape 错误：{fbank_path}\n"
                f"当前={tuple(fbank.shape)}，"
                f"要求={self.expected_shape}"
            )

        if not np.isfinite(fbank).all():
            raise ValueError(
                f"Fbank 包含 NaN 或 Inf："
                f"{fbank_path}"
            )

        x = torch.from_numpy(
            fbank
        ).float()

        if (
            self.training
            and self.cfg["USE_SPECAUGMENT"]
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

        # [T, F] -> [1, T, F]
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
        loader_args["prefetch_factor"] = 2

    return DataLoader(
        **loader_args
    )


# ============================================================
# 9. 纯 Fbank Patch Embedding
# ============================================================
class FbankPatchEmbedding(nn.Module):
    """
    不加载任何 AST 权重。

    输入：
        [B, 1, 798, 128]

    转换：
        [B, 1, 128, 798]

    Conv2d：
        kernel = 16 × 16
        stride = 10 × 10

    输出：
        [B, 256, 12, 79]

    Flatten：
        [B, 948, 256]
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=(16, 16),
        stride=(10, 10),
        freq_patches=12,
        time_patches=79,
        dropout=0.15,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.freq_patches = freq_patches
        self.time_patches = time_patches

        self.num_patches = (
            freq_patches
            * time_patches
        )

        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.projection.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if self.projection.bias is not None:
            nn.init.zeros_(
                self.projection.bias
            )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                "输入必须为 [B,1,T,F]，"
                f"当前为 {tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"输入通道必须为1，当前为{x.shape[1]}"
            )

        # [B,1,T,F] -> [B,1,F,T]
        x = x.transpose(
            2,
            3,
        ).contiguous()

        # [B,1,128,798]
        # -> [B,256,12,79]
        x = self.projection(x)

        if tuple(x.shape[2:]) != (
            self.freq_patches,
            self.time_patches,
        ):
            raise RuntimeError(
                "Patch Embedding 输出尺寸错误："
                f"当前={tuple(x.shape)}，"
                f"要求空间尺寸="
                f"{(self.freq_patches, self.time_patches)}"
            )

        # [B,256,12,79]
        # -> [B,256,948]
        # -> [B,948,256]
        x = x.flatten(2).transpose(
            1,
            2,
        ).contiguous()

        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


# ============================================================
# 10. 纯 Fbank 模型
# ============================================================
class PureFbankModel(nn.Module):
    """
    Fbank
      ↓
    随机初始化 Patch Embedding
      ↓
    Time-Mamba
      ↓
    Frequency-Attention
      ↓
    四分类
    """

    def __init__(self, cfg):
        super().__init__()

        self.patch_embedding = (
            FbankPatchEmbedding(
                embed_dim=cfg[
                    "PATCH_DIM"
                ],
                kernel_size=cfg[
                    "PATCH_KERNEL"
                ],
                stride=cfg[
                    "PATCH_STRIDE"
                ],
                freq_patches=cfg[
                    "FREQ_PATCHES"
                ],
                time_patches=cfg[
                    "TIME_PATCHES"
                ],
                dropout=cfg[
                    "DROPOUT"
                ],
            )
        )

        self.backbone = TimeFrequencyEncoder(
            input_dim=cfg["PATCH_DIM"],
            d_model=cfg["D_MODEL"],
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
            num_heads=cfg["NHEAD"],
            dropout=cfg["DROPOUT"],
        )

        self.head = nn.Sequential(
            nn.LayerNorm(
                cfg["D_MODEL"]
            ),
            nn.Dropout(
                cfg["HEAD_DROPOUT"]
            ),
            nn.Linear(
                cfg["D_MODEL"],
                4,
            ),
        )

    def extract_tokens(self, x):
        return self.patch_embedding(x)

    def forward(self, x):
        tokens = self.patch_embedding(x)

        feature = self.backbone(
            tokens
        )

        logits = self.head(
            feature
        )

        return logits


# ============================================================
# 11. 类别权重
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
        / np.maximum(counts, 1.0),
        cfg["FOUR_WEIGHT_POWER"],
    )

    weights[0] = 1.0

    weights = np.clip(
        weights,
        1.0,
        cfg["FOUR_WEIGHT_MAX"],
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
        / np.maximum(counts, 1.0),
        cfg["SUBTYPE_WEIGHT_POWER"],
    )

    weights = np.clip(
        weights,
        1.0,
        cfg["SUBTYPE_WEIGHT_MAX"],
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# 12. 联合损失
# ============================================================
def calculate_loss(
    logits,
    labels,
    four_weights,
    subtype_weights,
    cfg,
):
    # 四分类主损失
    four_loss = F.cross_entropy(
        logits,
        labels,
        weight=four_weights,
        label_smoothing=cfg[
            "LABEL_SMOOTHING"
        ],
    )

    # Normal / Abnormal 辅助损失
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

    # Crackle / Wheeze / Both 辅助损失
    abnormal_mask = labels > 0

    if abnormal_mask.any():
        subtype_logits = logits[
            abnormal_mask,
            1:4,
        ]

        subtype_target = (
            labels[abnormal_mask] - 1
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
        +
        cfg["BINARY_LOSS_WEIGHT"]
        * binary_loss
        +
        cfg["SUBTYPE_LOSS_WEIGHT"]
        * subtype_loss
    )

    return {
        "total": total_loss,
        "four": four_loss,
        "binary": binary_loss,
        "subtype": subtype_loss,
    }


# ============================================================
# 13. 训练一个 Epoch
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
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    total_batches = len(loader)
    epoch_start = time.time()

    print(
        f"[TRAIN] batches={total_batches} | "
        f"batch={cfg['BATCH_SIZE']} | "
        f"accum={cfg['ACCUM_STEPS']}",
        flush=True,
    )

    for batch_index, (x, y) in enumerate(loader):
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

            losses = calculate_loss(
                logits,
                y,
                four_weights,
                subtype_weights,
                cfg,
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

        should_step = (
            (batch_index + 1)
            % cfg["ACCUM_STEPS"]
            == 0
            or
            batch_index + 1
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

        completed = batch_index + 1

        if (
            completed == 1
            or
            completed
            % cfg["PRINT_INTERVAL"]
            == 0
            or
            completed == total_batches
        ):
            elapsed = (
                time.time()
                - epoch_start
            )

            average_batch_time = (
                elapsed / completed
            )

            eta_seconds = (
                total_batches - completed
            ) * average_batch_time

            if device.type == "cuda":
                allocated = (
                    torch.cuda
                    .memory_allocated(device)
                    / 1024 ** 3
                )

                reserved = (
                    torch.cuda
                    .memory_reserved(device)
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
                f"Bin "
                f"{losses['binary'].item():.4f} | "
                f"Sub "
                f"{losses['subtype'].item():.4f} | "
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
        for key, value in sums.items()
    }


# ============================================================
# 14. ICBHI 指标
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
        int(cm[0].sum()),
        1,
    )

    abnormal_total = max(
        int(cm[1:].sum()),
        1,
    )

    specificity = (
        100.0
        * float(cm[0, 0])
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
    ).astype(np.int64)

    binary_pred = (
        y_pred > 0
    ).astype(np.int64)

    return {
        "score": float(score),
        "sp": float(specificity),
        "se": float(sensitivity),

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
# 15. 官方测试集评估
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
# 16. Shape Test
# ============================================================
@torch.no_grad()
def shape_test(
    loader,
    model,
    device,
):
    model.eval()

    x, _ = next(iter(loader))

    x = x[:1].to(device)

    tokens = model.extract_tokens(
        x
    )

    logits = model(x)

    print(
        "[Shape] Fbank:",
        tuple(x.shape),
    )

    print(
        "[Shape] Fbank patch features:",
        tuple(tokens.shape),
    )

    print(
        "[Shape] logits:",
        tuple(logits.shape),
    )

    if tuple(x.shape) != (
        1,
        1,
        798,
        128,
    ):
        raise RuntimeError(
            f"Fbank shape 错误："
            f"{tuple(x.shape)}"
        )

    if tuple(tokens.shape) != (
        1,
        948,
        256,
    ):
        raise RuntimeError(
            "Fbank Patch 特征 shape 错误："
            f"{tuple(tokens.shape)}"
        )

    if tuple(logits.shape) != (
        1,
        4,
    ):
        raise RuntimeError(
            f"Logits shape 错误："
            f"{tuple(logits.shape)}"
        )

    model.train()


# ============================================================
# 17. 输出最终结果
# ============================================================
def print_final(result):
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
        result["four_cm"]
    )

    print()
    print(
        "Binary confusion matrix:"
    )
    print(
        result["binary_cm"]
    )


# ============================================================
# 18. 主函数
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
            torch.cuda.get_device_name(0),
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
            "mamba_ssm 导入失败。"
        )

    root = Path(
        cfg["ROOT"]
    )

    train_csv = (
        root / "train_index.csv"
    )

    test_csv = (
        root / "test_index.csv"
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
        "[Protocol] train_index.csv："
        "完整官方训练集"
    )

    print(
        "[Protocol] test_index.csv："
        "官方测试集"
    )

    print(
        "[Protocol] 不划验证集，"
        "固定训练轮数，最后测试一次。"
    )

    print(
        "[Input] 直接读取 Fbank，"
        "不读取 tokens_path，"
        "不加载任何 AST 模型或权重。"
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
        / "last_pure_fbank_model.pth"
    )

    final_path = (
        save_dir
        / "final_pure_fbank_model.pth"
    )

    history_path = (
        save_dir
        / "training_history.csv"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = PureFbankModel(
        cfg
    ).to(device)

    shape_test(
        train_loader,
        model,
        device,
    )

    # --------------------------------------------------------
    # 类别权重
    # --------------------------------------------------------
    four_weights = build_four_weights(
        train_set.class_counts,
        cfg,
    ).to(device)

    subtype_weights = (
        build_subtype_weights(
            train_set.class_counts,
            cfg,
        ).to(device)
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

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model
                    .patch_embedding
                    .parameters()
                ),
                "lr": cfg["PATCH_LR"],
            },
            {
                "params": (
                    model
                    .backbone
                    .parameters()
                ),
                "lr": cfg["BACKBONE_LR"],
            },
            {
                "params": (
                    model
                    .head
                    .parameters()
                ),
                "lr": cfg["HEAD_LR"],
            },
        ],
        weight_decay=cfg[
            "WEIGHT_DECAY"
        ],
    )

    base_lrs = [
        cfg["PATCH_LR"],
        cfg["BACKBONE_LR"],
        cfg["HEAD_LR"],
    ]

    min_lrs = [
        cfg["MIN_PATCH_LR"],
        cfg["MIN_BACKBONE_LR"],
        cfg["MIN_HEAD_LR"],
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
        "PURE FBANK → TIME-MAMBA "
        "→ FREQUENCY-ATTENTION TRAINING"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # 固定轮数训练
    # --------------------------------------------------------
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
            total_epochs=cfg["EPOCHS"],
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
            "patch_lr": current_lrs[0],
            "backbone_lr": current_lrs[1],
            "head_lr": current_lrs[2],
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
                "config": deepcopy(cfg),
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
            f"Bin "
            f"{train_result['binary']:.4f} | "
            f"Sub "
            f"{train_result['subtype']:.4f} | "
            f"LR "
            f"{current_lrs[0]:.8f}/"
            f"{current_lrs[1]:.8f}/"
            f"{current_lrs[2]:.8f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    # --------------------------------------------------------
    # 最后只测试一次
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
            "epoch": cfg["EPOCHS"],
            "model_state": (
                model.state_dict()
            ),
            "config": deepcopy(cfg),

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