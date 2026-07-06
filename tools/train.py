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
from transformers import ASTModel


# ============================================================
# 1. 配置
# ============================================================
CONFIG = {
    # Fbank 数据目录
    "ROOT": "/data/dingcong/hybrid/icbhi_official_fbank",

    # 模型保存目录
    "SAVE_DIR": "/data/dingcong/hybrid/checkpoints_fbank_ast_projection",

    # 与 datapreprocess.py 使用相同的 AST
    "AST_MODEL": "MIT/ast-finetuned-audioset-10-10-0.4593",

    # 模型已下载到 HuggingFace 缓存时设为 True
    "LOCAL_FILES_ONLY": True,

    # 是否冻结 AST Patch Projection
    # False：允许预训练 Patch Projection 跟随任务微调
    "FREEZE_PATCH_PROJECTION": False,

    # DataLoader
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 8,
    "NUM_WORKERS": 1,

    # 固定训练轮数，不使用验证集
    "EPOCHS": 50,

    "SEED": 42,
    "DEVICE": "cuda",
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # Fbank
    "FBANK_FRAMES": 798,
    "FBANK_MELS": 128,

    # AST Patch Projection 输出
    "INPUT_DIM": 768,
    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    # 你原有的 Time-Mamba + Frequency-Attention
    "D_MODEL": 256,
    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,
    "NHEAD": 8,
    "DROPOUT": 0.15,

    # 分类头
    "HEAD_DROPOUT": 0.20,

    # 学习率
    "PATCH_LR": 1e-5,
    "BACKBONE_LR": 1e-5,
    "HEAD_LR": 5e-5,

    "MIN_PATCH_LR": 1e-6,
    "MIN_BACKBONE_LR": 1e-6,
    "MIN_HEAD_LR": 5e-6,

    "WARMUP_EPOCHS": 3,
    "WEIGHT_DECAY": 1e-2,
    "GRAD_CLIP": 2.0,

    # 联合损失
    "FOUR_LOSS_WEIGHT": 1.0,
    "BINARY_LOSS_WEIGHT": 0.05,
    "SUBTYPE_LOSS_WEIGHT": 1.0,
    "LABEL_SMOOTHING": 0.0,

    # 类别权重
    "FOUR_WEIGHT_POWER": 0.50,
    "FOUR_WEIGHT_MAX": 2.20,

    "SUBTYPE_WEIGHT_POWER": 0.50,
    "SUBTYPE_WEIGHT_MAX": 2.00,
}


# ============================================================
# 2. 导入你原来的模型
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import HAS_MAMBA, TimeFrequencyEncoder


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
# 5. 加载 checkpoint
# ============================================================
def safe_load(path: Path, device: torch.device):
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
# 6. Warmup + Cosine 学习率
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
        scale = 0.2 + 0.8 * epoch / max(warmup_epochs, 1)

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
            + (base_lr - min_lr) * cosine_ratio
            for base_lr, min_lr
            in zip(base_lrs, min_lrs)
        ]

    for group, lr in zip(
        optimizer.param_groups,
        current_lrs,
    ):
        group["lr"] = float(lr)

    return current_lrs


# ============================================================
# 7. Fbank Dataset
# ============================================================
class FbankDataset(Dataset):
    """
    CSV 中读取：
        fbank_path
        label

    单个 npy：
        [798, 128] = [T, F]

    返回：
        x: [1, 798, 128]
        y: scalar
    """

    def __init__(
        self,
        csv_path,
        expected_shape=(798, 128),
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.expected_shape = tuple(expected_shape)

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

        missing = required_columns - set(
            self.df.columns
        )

        if missing:
            raise ValueError(
                f"{self.csv_path} 缺少列："
                f"{sorted(missing)}"
            )

        self.df["label"] = (
            self.df["label"].astype(int)
        )

        self.labels = self.df[
            "label"
        ].to_numpy(dtype=np.int64)

        invalid = np.unique(
            self.labels[
                (self.labels < 0)
                | (self.labels > 3)
            ]
        )

        if len(invalid) > 0:
            raise ValueError(
                f"发现非法标签：{invalid.tolist()}"
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
            f"csv={self.csv_path}"
        )

    def __len__(self):
        return len(self.df)

    def resolve_path(self, raw_path):
        path = Path(str(raw_path))

        if path.exists():
            return path

        relative_path = self.csv_path.parent / path

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
                f"当前：{tuple(fbank.shape)}\n"
                f"要求：{self.expected_shape}"
            )

        if not np.isfinite(fbank).all():
            raise ValueError(
                f"Fbank 包含 NaN 或 Inf：{fbank_path}"
            )

        # [T, F] -> [1, T, F]
        x = torch.from_numpy(
            fbank
        ).float().unsqueeze(0)

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
    workers = int(cfg["NUM_WORKERS"])

    return DataLoader(
        dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        drop_last=False,
    )


# ============================================================
# 9. Fbank -> AST Patch Projection -> 原有 Hybrid
# ============================================================
class FbankHybridModel(nn.Module):
    """
    输入：
        [B, 1, 798, 128]

    在线 AST Patch Projection：
        [B, 1, 128, 798]
              ↓
        Conv2d(1, 768, kernel=16, stride=10)
              ↓
        [B, 768, 12, 79]
              ↓
        [B, 948, 768]

    然后进入原来的：
        Time-Mamba
              ↓
        Frequency-Attention
              ↓
        Pooling
              ↓
        四分类
    """

    def __init__(self, cfg):
        super().__init__()

        self.fbank_frames = int(
            cfg["FBANK_FRAMES"]
        )

        self.fbank_mels = int(
            cfg["FBANK_MELS"]
        )

        self.freq_patches = int(
            cfg["FREQ_PATCHES"]
        )

        self.time_patches = int(
            cfg["TIME_PATCHES"]
        )

        self.num_patches = (
            self.freq_patches
            * self.time_patches
        )

        print(
            "[INIT] 加载预训练 AST Patch Projection：",
            cfg["AST_MODEL"],
        )

        ast_model = ASTModel.from_pretrained(
            cfg["AST_MODEL"],
            local_files_only=cfg[
                "LOCAL_FILES_ONLY"
            ],
        )

        # 只保留 AST 的 Patch Projection
        self.patch_projection = deepcopy(
            ast_model
            .embeddings
            .patch_embeddings
            .projection
        )

        del ast_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg["FREEZE_PATCH_PROJECTION"]:
            for parameter in (
                self.patch_projection.parameters()
            ):
                parameter.requires_grad = False

        self.backbone = TimeFrequencyEncoder(
            input_dim=cfg["INPUT_DIM"],
            d_model=cfg["D_MODEL"],
            freq_patches=cfg["FREQ_PATCHES"],
            time_patches=cfg["TIME_PATCHES"],
            time_depth=cfg["TIME_DEPTH"],
            freq_depth=cfg["FREQ_DEPTH"],
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

    def extract_patch_tokens(self, x):
        if x.ndim != 4:
            raise ValueError(
                "输入必须为 [B, 1, T, F]，"
                f"当前为 {tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"输入通道必须为1，当前为{x.shape[1]}"
            )

        if tuple(x.shape[2:]) != (
            self.fbank_frames,
            self.fbank_mels,
        ):
            raise ValueError(
                "Fbank shape 错误："
                f"当前={tuple(x.shape[2:])}，"
                f"要求={(self.fbank_frames, self.fbank_mels)}"
            )

        # [B, 1, T, F] -> [B, 1, F, T]
        x = x.transpose(
            2,
            3,
        ).contiguous()

        # [B, 1, 128, 798]
        # -> [B, 768, 12, 79]
        x = self.patch_projection(x)

        if tuple(x.shape[2:]) != (
            self.freq_patches,
            self.time_patches,
        ):
            raise RuntimeError(
                "AST Patch Projection 输出尺寸错误："
                f"当前={tuple(x.shape)}，"
                f"要求空间尺寸="
                f"{(self.freq_patches, self.time_patches)}"
            )

        # [B, 768, 12, 79]
        # -> [B, 768, 948]
        # -> [B, 948, 768]
        tokens = x.flatten(2).transpose(
            1,
            2,
        ).contiguous()

        if tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"Token数量错误：{tokens.shape[1]}，"
                f"要求：{self.num_patches}"
            )

        return tokens

    def forward(self, x):
        tokens = self.extract_patch_tokens(x)

        feature = self.backbone(tokens)

        logits = self.head(feature)

        return logits


# ============================================================
# 10. 四分类类别权重
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


# ============================================================
# 11. 异常子类权重
# ============================================================
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

    # Normal / Abnormal 二分类辅助损失
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
        + cfg["BINARY_LOSS_WEIGHT"]
        * binary_loss
        + cfg["SUBTYPE_LOSS_WEIGHT"]
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

            loss = (
                losses["total"]
                / cfg["ACCUM_STEPS"]
            )

        scaler.scale(loss).backward()

        for key in sums:
            sums[key] += float(
                losses[key].detach().item()
            )

        should_step = (
            (batch_index + 1)
            % cfg["ACCUM_STEPS"]
            == 0
            or
            batch_index + 1
            == len(loader)
        )

        if should_step:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                cfg["GRAD_CLIP"],
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

    divisor = max(len(loader), 1)

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
        specificity + sensitivity
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
            )
            * 100.0
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

        prediction = torch.argmax(
            logits,
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

    x, y = next(iter(loader))

    x_one = x[:1].to(device)

    tokens = model.extract_patch_tokens(
        x_one
    )

    logits = model(x_one)

    print(
        "[Shape] Fbank:",
        tuple(x_one.shape),
    )

    print(
        "[Shape] AST tokens:",
        tuple(tokens.shape),
    )

    print(
        "[Shape] logits:",
        tuple(logits.shape),
    )

    if tuple(x_one.shape) != (
        1,
        1,
        798,
        128,
    ):
        raise RuntimeError(
            f"Fbank shape 错误：{tuple(x_one.shape)}"
        )

    if tuple(tokens.shape) != (
        1,
        948,
        768,
    ):
        raise RuntimeError(
            f"AST tokens shape 错误："
            f"{tuple(tokens.shape)}"
        )

    if tuple(logits.shape) != (
        1,
        4,
    ):
        raise RuntimeError(
            f"logits shape 错误："
            f"{tuple(logits.shape)}"
        )

    model.train()


# ============================================================
# 17. 打印最终结果
# ============================================================
def print_final(result):
    print()
    print("=" * 80)
    print("FINAL OFFICIAL TEST RESULT")
    print("=" * 80)

    print(
        f"ICBHI Score: {result['score']:.4f}"
    )

    print(
        f"Specificity: {result['sp']:.4f}"
    )

    print(
        f"Sensitivity: {result['se']:.4f}"
    )

    print(
        f"Accuracy: {result['accuracy']:.4f}"
    )

    print(
        f"Macro-F1: {result['macro_f1']:.4f}"
    )

    print(
        "Recall [Normal, Crackle, Wheeze, Both]:",
        np.round(
            result["recalls"],
            4,
        ).tolist(),
    )

    print(
        "PredCount:",
        result["pred_counts"].tolist(),
    )

    print()
    print("Four-class confusion matrix:")
    print(result["four_cm"])

    print()
    print("Binary confusion matrix:")
    print(result["binary_cm"])


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
            "mamba_ssm 导入失败，不能进行正式训练。"
        )

    root = Path(
        cfg["ROOT"]
    )

    train_csv = root / "train_index.csv"
    test_csv = root / "test_index.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            train_csv
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            test_csv
        )

    print(
        "[Protocol] 官方训练集：train_index.csv"
    )

    print(
        "[Protocol] 官方测试集：test_index.csv"
    )

    print(
        "[Protocol] 不划分验证集，固定训练轮数，"
        "训练结束后测试一次。"
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
        / "last_fbank_model.pth"
    )

    final_path = (
        save_dir
        / "final_fbank_model.pth"
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
        expected_shape=(
            cfg["FBANK_FRAMES"],
            cfg["FBANK_MELS"],
        ),
    )

    test_set = FbankDataset(
        test_csv,
        expected_shape=(
            cfg["FBANK_FRAMES"],
            cfg["FBANK_MELS"],
        ),
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
    model = FbankHybridModel(
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

    subtype_weights = build_subtype_weights(
        train_set.class_counts,
        cfg,
    ).to(device)

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
    patch_parameters = [
        parameter
        for parameter
        in model.patch_projection.parameters()
        if parameter.requires_grad
    ]

    parameter_groups = []
    base_lrs = []
    min_lrs = []

    if patch_parameters:
        parameter_groups.append({
            "params": patch_parameters,
            "lr": cfg["PATCH_LR"],
        })

        base_lrs.append(
            cfg["PATCH_LR"]
        )

        min_lrs.append(
            cfg["MIN_PATCH_LR"]
        )

    parameter_groups.append({
        "params": model.backbone.parameters(),
        "lr": cfg["BACKBONE_LR"],
    })

    base_lrs.append(
        cfg["BACKBONE_LR"]
    )

    min_lrs.append(
        cfg["MIN_BACKBONE_LR"]
    )

    parameter_groups.append({
        "params": model.head.parameters(),
        "lr": cfg["HEAD_LR"],
    })

    base_lrs.append(
        cfg["HEAD_LR"]
    )

    min_lrs.append(
        cfg["MIN_HEAD_LR"]
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=cfg[
            "WEIGHT_DECAY"
        ],
    )

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
    print("FBANK ONLINE AST-PROJECTION TRAINING")
    print("=" * 90)

    # --------------------------------------------------------
    # 固定轮数训练
    # --------------------------------------------------------
    for epoch in range(
        1,
        cfg["EPOCHS"] + 1,
    ):
        start_time = time.time()

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

        elapsed = time.time() - start_time

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
            "seconds": elapsed,
        }

        for index, lr in enumerate(
            current_lrs
        ):
            history_row[
                f"lr_group_{index}"
            ] = lr

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
                "four_weights": (
                    four_weights
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "subtype_weights": (
                    subtype_weights
                    .detach()
                    .cpu()
                    .tolist()
                ),
            },
            last_path,
        )

        lr_text = "/".join(
            f"{lr:.8f}"
            for lr in current_lrs
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{cfg['EPOCHS']} | "
            f"Train {train_result['total']:.4f} | "
            f"Four {train_result['four']:.4f} | "
            f"Bin {train_result['binary']:.4f} | "
            f"Sub {train_result['subtype']:.4f} | "
            f"LR {lr_text} | "
            f"{elapsed:.1f}s"
        )

    # --------------------------------------------------------
    # 训练完成后，官方测试集只评估一次
    # --------------------------------------------------------
    checkpoint = safe_load(
        last_path,
        device,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

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
            "epoch": checkpoint["epoch"],
            "model_state": model.state_dict(),
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