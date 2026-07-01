#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. 配置
# ============================================================
CONFIG = {
    # 数据目录
    # 目录中需要包含：
    #   train_index.csv
    #   test_index.csv
    "ROOT": "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",

    # 训练参数
    "EPOCHS": 30,
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,

    "LR": 1e-5,
    "WEIGHT_DECAY": 1e-2,
    "NUM_WORKERS": 1,
    "SEED": 42,
    "DEVICE": "cuda",

    # 使用新目录，避免覆盖上一轮结果
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_serial_tmamba_fattention_v2"
    ),

    "PATIENCE": 8,

    # 模型参数
    # 948 = 12 × 79
    "TOKEN_DIM": 768,
    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 2,
    "FREQ_DEPTH": 2,
    "NHEAD": 8,

    # 模型内部 Dropout
    "DROPOUT": 0.2,

    # 分类头 Dropout
    "CLASSIFIER_DROPOUT": 0.2,

    # 数据增强
    # 这一轮继续关闭，避免同时改变过多变量
    "SPEC_AUG": False,
    "MAX_MASK_T": 8,
    "MAX_MASK_F": 2,
    "NUM_MASKS": 2,

    # 损失函数
    "WEIGHTED_LOSS": True,
    "LABEL_SMOOTHING": 0.05,

    # AMP
    "AMP": True,
    "REQUIRE_MAMBA": True,

    # False：
    # 严格四分类评价，异常类别必须预测正确才计入 SE
    #
    # True：
    # Normal / Abnormal 二分类式评价
    "TWO_CLS_EVAL": False,
}


# ============================================================
# 2. 导入项目模型
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import HAS_MAMBA, TimeFrequencyEncoder


# ============================================================
# 3. 设置随机种子
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. Patch 级 SpecAugment
#
# 输入：
#   [948, 768]
#
# 恢复为：
#   [12, 79, 768]
#
# 时间遮挡作用于 79 个时间 Patch
# 频率遮挡作用于 12 个频率 Patch
# ============================================================
def apply_patch_augment(
    x: torch.Tensor,
    freq_patches: int,
    time_patches: int,
    max_mask_t: int = 8,
    max_mask_f: int = 2,
    num_masks: int = 2,
) -> torch.Tensor:

    expected_tokens = freq_patches * time_patches

    if x.ndim != 2:
        raise ValueError(
            f"SpecAugment 输入必须是 [N, D]，"
            f"当前形状为 {tuple(x.shape)}"
        )

    if x.shape[0] != expected_tokens:
        raise ValueError(
            f"Token 数量错误：当前为 {x.shape[0]}，"
            f"要求为 {expected_tokens}"
        )

    feature_dim = x.shape[-1]

    # [948, 768] -> [12, 79, 768]
    x_aug = x.reshape(
        freq_patches,
        time_patches,
        feature_dim,
    ).clone()

    for _ in range(num_masks):

        # 时间 Patch 遮挡
        t_width = random.randint(
            0,
            min(max_mask_t, time_patches),
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

        # 频率 Patch 遮挡
        f_width = random.randint(
            0,
            min(max_mask_f, freq_patches),
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
# 5. Dataset
# ============================================================
class TokenNPY4ClsDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        is_train: bool,
        specaug: bool,
        freq_patches: int,
        time_patches: int,
        token_dim: int,
        max_mask_t: int,
        max_mask_f: int,
        num_masks: int,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.df = pd.read_csv(
            csv_path
        ).reset_index(drop=True)

        required_columns = {
            "tokens_path",
            "label",
        }

        missing_columns = (
            required_columns - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{csv_path} 缺少列："
                f"{sorted(missing_columns)}；"
                f"当前列为："
                f"{self.df.columns.tolist()}"
            )

        self.is_train = is_train
        self.specaug = specaug

        self.freq_patches = freq_patches
        self.time_patches = time_patches
        self.token_dim = token_dim

        self.expected_tokens = (
            freq_patches * time_patches
        )

        self.max_mask_t = max_mask_t
        self.max_mask_f = max_mask_f
        self.num_masks = num_masks

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
                "四分类标签只能为 0、1、2、3；"
                f"发现无效标签："
                f"{invalid_labels.tolist()}"
            )

        self.class_counts_4 = np.bincount(
            self.labels,
            minlength=4,
        )

        print(
            f"[Dataset] Loaded {len(self.df)} samples "
            f"from {csv_path}"
        )

        print(
            f"[Dataset] class counts: "
            f"{self.class_counts_4.tolist()}"
        )

        print(
            f"[Dataset] train={self.is_train}, "
            f"specaug={self.specaug}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        row = self.df.iloc[index]

        token_path = str(
            row["tokens_path"]
        )

        if not os.path.exists(token_path):
            raise FileNotFoundError(
                f"Token 文件不存在：{token_path}"
            )

        tokens_np = np.load(
            token_path
        )

        expected_shape = (
            self.expected_tokens,
            self.token_dim,
        )

        if tuple(tokens_np.shape) != expected_shape:
            raise ValueError(
                f"Token shape error：{token_path}\n"
                f"当前形状：{tuple(tokens_np.shape)}\n"
                f"要求形状：{expected_shape}\n"
                f"其中 {self.expected_tokens} = "
                f"{self.freq_patches} × "
                f"{self.time_patches}"
            )

        x = torch.from_numpy(
            tokens_np
        ).float()

        if self.is_train and self.specaug:
            x = apply_patch_augment(
                x=x,
                freq_patches=self.freq_patches,
                time_patches=self.time_patches,
                max_mask_t=self.max_mask_t,
                max_mask_f=self.max_mask_f,
                num_masks=self.num_masks,
            )

        label = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long,
        )

        return x, label


# ============================================================
# 6. 固定形状 Collate
#
# 所有输入固定为：
#   [948, 768]
#
# 不再进行 Padding，也不再生成 mask
# ============================================================
def collate_fixed(
    batch: List[
        Tuple[torch.Tensor, torch.Tensor]
    ],
) -> Tuple[torch.Tensor, torch.Tensor]:

    xs, ys = zip(*batch)

    x_batch = torch.stack(
        xs,
        dim=0,
    )

    y_batch = torch.stack(
        ys,
        dim=0,
    ).view(-1)

    return x_batch, y_batch


# ============================================================
# 7. ICBHI Score
# ============================================================
def get_score_from_hits_counts(
    hits: List[float],
    counts: List[float],
) -> Tuple[float, float, float]:

    eps = 1e-10

    # Specificity：Normal 类正确率
    sp = 100.0 * (
        hits[0]
        / (counts[0] + eps)
    )

    # Sensitivity：三个异常类别整体正确率
    abnormal_hits = float(
        hits[1]
        + hits[2]
        + hits[3]
    )

    abnormal_counts = float(
        counts[1]
        + counts[2]
        + counts[3]
    )

    se = 100.0 * (
        abnormal_hits
        / (abnormal_counts + eps)
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
# 8. 验证
# ============================================================
@torch.no_grad()
def evaluate_like_author(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    two_cls_eval: bool,
) -> Dict[str, object]:

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

    loss_sum = 0.0
    sample_count = 0

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

        feature = backbone(x)
        logits = classifier(feature)

        loss = criterion(
            logits,
            y,
        )

        loss_sum += float(
            loss.item()
        )

        sample_count += int(
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

            if not two_cls_eval:
                # 严格四分类
                if pr == gt:
                    hits[gt] += 1.0

            else:
                # Normal / Abnormal 式评价
                if gt == 0 and pr == 0:
                    hits[gt] += 1.0

                elif gt != 0 and pr > 0:
                    hits[gt] += 1.0

    sp, se, score = (
        get_score_from_hits_counts(
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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
    )

    return {
        "SP": float(sp),
        "SE": float(se),
        "ICBHI": float(score),
        "ACC": float(accuracy),
        "F1": float(macro_f1),
        "LOSS": float(
            loss_sum
            / max(1, sample_count)
        ),
        "hits": hits,
        "counts": counts,
        "cm": cm,
    }


# ============================================================
# 9. 训练一个 Epoch
# ============================================================
def train_one_epoch(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
    accum_steps: int,
    class_weights: Optional[
        torch.Tensor
    ] = None,
    label_smoothing: float = 0.0,
) -> Tuple[float, int]:

    backbone.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss(
        weight=(
            class_weights.to(device)
            if class_weights is not None
            else None
        ),
        label_smoothing=label_smoothing,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    number_of_batches = len(loader)
    optimizer_steps = 0

    parameters = (
        list(backbone.parameters())
        + list(classifier.parameters())
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
            feature = backbone(x)
            logits = classifier(feature)

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

        running_loss += float(
            raw_loss.detach().item()
        )

        is_last_batch = (
            batch_index + 1
            == number_of_batches
        )

        should_step = (
            (batch_index + 1)
            % accum_steps
            == 0
            or is_last_batch
        )

        if should_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=5.0,
            )

            old_scale = scaler.get_scale()

            scaler.step(
                optimizer
            )

            scaler.update()

            new_scale = scaler.get_scale()

            # 如果 scale 没有下降，
            # 说明本次 optimizer.step() 没有因溢出被跳过
            if new_scale >= old_scale:
                optimizer_steps += 1

            optimizer.zero_grad(
                set_to_none=True
            )

    mean_loss = (
        running_loss
        / max(1, number_of_batches)
    )

    return (
        mean_loss,
        optimizer_steps,
    )


# ============================================================
# 10. AMP GradScaler
# ============================================================
def create_grad_scaler(
    use_amp: bool,
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
# 11. 模型形状检查
# ============================================================
@torch.no_grad()
def run_shape_test(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    device: torch.device,
) -> None:

    backbone.eval()
    classifier.eval()

    x, _ = next(
        iter(loader)
    )

    # 只使用一个样本，减少显存占用
    x = x[:1].to(
        device
    )

    feature = backbone(x)
    logits = classifier(feature)

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

    expected_feature_shape = (
        1,
        int(CONFIG["TOKEN_DIM"]),
    )

    expected_logits_shape = (
        1,
        4,
    )

    if tuple(feature.shape) != expected_feature_shape:
        raise RuntimeError(
            f"Backbone 输出形状错误："
            f"{tuple(feature.shape)}；"
            f"要求："
            f"{expected_feature_shape}"
        )

    if tuple(logits.shape) != expected_logits_shape:
        raise RuntimeError(
            f"Classifier 输出形状错误："
            f"{tuple(logits.shape)}；"
            f"要求："
            f"{expected_logits_shape}"
        )

    print(
        "[SHAPE TEST] Passed."
    )


# ============================================================
# 12. 主函数
# ============================================================
def main() -> None:

    cfg = CONFIG

    set_seed(
        int(cfg["SEED"])
    )

    # --------------------------------------------------------
    # 设备
    # --------------------------------------------------------
    use_cuda = (
        cfg["DEVICE"] == "cuda"
        and torch.cuda.is_available()
    )

    device = torch.device(
        "cuda"
        if use_cuda
        else "cpu"
    )

    print(
        f"[INFO] device: {device}"
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

    # --------------------------------------------------------
    # 检查 Mamba
    # --------------------------------------------------------
    print(
        f"[INFO] HAS_MAMBA = "
        f"{HAS_MAMBA}"
    )

    if (
        cfg["REQUIRE_MAMBA"]
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm 没有成功导入。"
            "请先确认当前 Conda 环境和 "
            "mamba_ssm 安装是否正确。"
        )

    # --------------------------------------------------------
    # 数据文件
    # --------------------------------------------------------
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
            f"找不到训练 CSV："
            f"{train_csv}"
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            f"找不到测试 CSV："
            f"{test_csv}"
        )

    # --------------------------------------------------------
    # 保存目录
    # --------------------------------------------------------
    os.makedirs(
        cfg["SAVE_DIR"],
        exist_ok=True,
    )

    best_checkpoint_path = os.path.join(
        cfg["SAVE_DIR"],
        "best.pth",
    )

    last_checkpoint_path = os.path.join(
        cfg["SAVE_DIR"],
        "last.pth",
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset = TokenNPY4ClsDataset(
        csv_path=str(
            train_csv
        ),
        is_train=True,
        specaug=bool(
            cfg["SPEC_AUG"]
        ),
        freq_patches=int(
            cfg["FREQ_PATCHES"]
        ),
        time_patches=int(
            cfg["TIME_PATCHES"]
        ),
        token_dim=int(
            cfg["TOKEN_DIM"]
        ),
        max_mask_t=int(
            cfg["MAX_MASK_T"]
        ),
        max_mask_f=int(
            cfg["MAX_MASK_F"]
        ),
        num_masks=int(
            cfg["NUM_MASKS"]
        ),
    )

    validation_dataset = TokenNPY4ClsDataset(
        csv_path=str(
            test_csv
        ),
        is_train=False,
        specaug=False,
        freq_patches=int(
            cfg["FREQ_PATCHES"]
        ),
        time_patches=int(
            cfg["TIME_PATCHES"]
        ),
        token_dim=int(
            cfg["TOKEN_DIM"]
        ),
        max_mask_t=0,
        max_mask_f=0,
        num_masks=0,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            cfg["BATCH_SIZE"]
        ),
        shuffle=True,
        num_workers=int(
            cfg["NUM_WORKERS"]
        ),
        pin_memory=pin_memory,
        collate_fn=collate_fixed,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(
            cfg["BATCH_SIZE"]
        ),
        shuffle=False,
        num_workers=int(
            cfg["NUM_WORKERS"]
        ),
        pin_memory=pin_memory,
        collate_fn=collate_fixed,
        drop_last=False,
    )

    # --------------------------------------------------------
    # Backbone
    #
    # Time-Mamba -> Frequency-Attention
    # --------------------------------------------------------
    backbone = TimeFrequencyEncoder(
        token_dim=int(
            cfg["TOKEN_DIM"]
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
    ).to(device)

    # 分类器
    classifier = nn.Sequential(
        nn.Dropout(
            float(
                cfg["CLASSIFIER_DROPOUT"]
            )
        ),
        nn.Linear(
            int(cfg["TOKEN_DIM"]),
            4,
        ),
    ).to(device)

    print(
        "[MODEL] Serial structure:"
    )

    print(
        "[MODEL] Time-Mamba "
        "-> Frequency-Attention "
        "-> Mean Pooling "
        "-> Dropout "
        "-> Linear Classifier"
    )

    print(
        "[MODEL] Input shape: "
        f"[B, "
        f"{cfg['FREQ_PATCHES'] * cfg['TIME_PATCHES']}, "
        f"{cfg['TOKEN_DIM']}]"
    )

    # --------------------------------------------------------
    # 形状检查
    # --------------------------------------------------------
    run_shape_test(
        loader=train_loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
    )

    # --------------------------------------------------------
    # 类别权重
    #
    # 使用平方根反频率权重：
    #   1 / sqrt(class_count)
    #
    # 比直接使用 1 / class_count 更温和
    # --------------------------------------------------------
    class_weights = None

    if cfg["WEIGHTED_LOSS"]:

        class_counts = (
            train_dataset
            .class_counts_4
            .astype(np.float32)
        )

        weights = (
            1.0
            / np.sqrt(
                np.maximum(
                    class_counts,
                    1.0,
                )
            )
        )

        # 平均权重归一化为 1
        weights = (
            weights
            / weights.mean()
        )

        class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
        )

        print(
            "[INFO] Weighted loss enabled."
        )

        print(
            "[INFO] Class counts:",
            class_counts.tolist(),
        )

        print(
            "[INFO] Class weights:",
            weights.tolist(),
        )

        print(
            "[INFO] Label smoothing:",
            float(
                cfg["LABEL_SMOOTHING"]
            ),
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    trainable_parameters = (
        list(backbone.parameters())
        + list(classifier.parameters())
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(
            cfg["LR"]
        ),
        weight_decay=float(
            cfg["WEIGHT_DECAY"]
        ),
        betas=(
            0.9,
            0.999,
        ),
    )

    # --------------------------------------------------------
    # Scheduler
    #
    # 每个 Epoch 更新一次，而不是每个 Batch 更新
    # 可以避免 AMP 跳过 optimizer.step 时产生顺序警告
    # --------------------------------------------------------
    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=int(
                cfg["EPOCHS"]
            ),
            eta_min=1e-6,
        )
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------
    use_amp = bool(
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

    # --------------------------------------------------------
    # 训练状态
    # --------------------------------------------------------
    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    print()
    print(
        "=" * 70
    )

    print(
        "Start training: "
        "Serial Time-Mamba "
        "-> Frequency-Attention"
    )

    print(
        "=" * 70
    )
    print()

    # --------------------------------------------------------
    # Epoch Loop
    # --------------------------------------------------------
    for epoch in range(
        1,
        int(cfg["EPOCHS"]) + 1,
    ):

        start_time = time.time()

        train_loss, optimizer_steps = train_one_epoch(
            loader=train_loader,
            backbone=backbone,
            classifier=classifier,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            accum_steps=int(
                cfg["ACCUM_STEPS"]
            ),
            class_weights=class_weights,
            label_smoothing=float(
                cfg["LABEL_SMOOTHING"]
            ),
        )

        validation_metrics = evaluate_like_author(
            loader=validation_loader,
            backbone=backbone,
            classifier=classifier,
            device=device,
            two_cls_eval=bool(
                cfg["TWO_CLS_EVAL"]
            ),
        )

        # 当前 Epoch 实际使用的学习率
        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # 每个 Epoch 更新一次 Scheduler
        if optimizer_steps > 0:
            scheduler.step()

        else:
            print(
                "[AMP WARNING] "
                "本 Epoch 没有完成有效的 "
                "optimizer.step()，"
                "scheduler 未更新。"
            )

        current_score = float(
            validation_metrics["ICBHI"]
        )

        improved = (
            current_score
            > best_score + 1e-9
        )

        # ----------------------------------------------------
        # 保存最佳模型
        # ----------------------------------------------------
        if improved:

            best_score = current_score
            best_epoch = epoch
            bad_epochs = 0
            marker = "BEST"

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
                    "best_score": best_score,
                    "config": deepcopy(cfg),
                },
                best_checkpoint_path,
            )

        else:
            bad_epochs += 1
            marker = "    "

        # ----------------------------------------------------
        # 保存最后一个 Epoch
        # ----------------------------------------------------
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
                "best_score": best_score,
                "config": deepcopy(cfg),
            },
            last_checkpoint_path,
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        print(
            f"[{marker}] "
            f"Epoch {epoch:03d}/"
            f"{cfg['EPOCHS']} | "
            f"train_loss "
            f"{train_loss:.4f} | "
            f"val_loss "
            f"{validation_metrics['LOSS']:.4f} | "
            f"Score "
            f"{validation_metrics['ICBHI']:.4f} | "
            f"SP "
            f"{validation_metrics['SP']:.4f} | "
            f"SE "
            f"{validation_metrics['SE']:.4f} | "
            f"ACC "
            f"{validation_metrics['ACC']:.4f} | "
            f"Macro-F1 "
            f"{validation_metrics['F1']:.4f} | "
            f"LR "
            f"{current_lr:.8f} | "
            f"OptSteps "
            f"{optimizer_steps} | "
            f"{elapsed_time:.1f}s"
        )

        # ----------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------
        if bad_epochs >= int(
            cfg["PATIENCE"]
        ):
            print(
                f"[EARLY STOP] "
                f"连续 {cfg['PATIENCE']} 个 Epoch "
                "Score 没有提升。"
            )

            print(
                f"[EARLY STOP] "
                f"best epoch = {best_epoch}, "
                f"best score = "
                f"{best_score:.4f}"
            )

            break

    # --------------------------------------------------------
    # 训练结束
    # --------------------------------------------------------
    print()
    print(
        "=" * 70
    )

    print(
        f"Training completed. "
        f"Best Score = "
        f"{best_score:.4f}, "
        f"Best Epoch = "
        f"{best_epoch}"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 加载最佳模型并最终评价
    # --------------------------------------------------------
    checkpoint = torch.load(
        best_checkpoint_path,
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

    final_metrics = evaluate_like_author(
        loader=validation_loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
        two_cls_eval=bool(
            cfg["TWO_CLS_EVAL"]
        ),
    )

    print()
    print(
        "[FINAL RESULT]"
    )

    print(
        f"Score: "
        f"{final_metrics['ICBHI']:.4f}"
    )

    print(
        f"SP: "
        f"{final_metrics['SP']:.4f}"
    )

    print(
        f"SE: "
        f"{final_metrics['SE']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{final_metrics['ACC']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{final_metrics['F1']:.4f}"
    )

    print()
    print(
        "[FINAL] Confusion Matrix"
    )

    print(
        "Rows = true labels 0/1/2/3"
    )

    print(
        "Columns = predicted labels 0/1/2/3"
    )

    print(
        final_metrics["cm"]
    )


if __name__ == "__main__":
    main()