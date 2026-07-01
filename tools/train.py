#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
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
    # --------------------------------------------------------
    # 数据目录
    # 目录中需要包含：
    #   train_index.csv
    #   test_index.csv
    #
    # CSV 至少包含：
    #   tokens_path
    #   label
    # --------------------------------------------------------
    "ROOT": "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",

    # --------------------------------------------------------
    # 训练参数
    # 第一次运行先设为 1，确认没有报错后再改成 30 或 50
    # --------------------------------------------------------
    "EPOCHS": 1,
    "BATCH_SIZE": 2,
    "ACCUM_STEPS": 8,

    "LR": 1e-5,
    "WEIGHT_DECAY": 1e-2,
    "NUM_WORKERS": 1,
    "SEED": 42,
    "DEVICE": "cuda",

    # 新目录，避免覆盖原来的并行模型
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_serial_tmamba_fattention"
    ),

    "PATIENCE": 10,

    # --------------------------------------------------------
    # 模型参数
    #
    # AST patch token:
    #   [948, 768]
    #
    # 948 = 12 × 79
    # --------------------------------------------------------
    "TOKEN_DIM": 768,
    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 2,
    "FREQ_DEPTH": 2,
    "NHEAD": 8,
    "DROPOUT": 0.1,

    # --------------------------------------------------------
    # 数据增强
    #
    # 当前先关闭，保证与原模型进行公平比较
    # --------------------------------------------------------
    "SPEC_AUG": False,
    "MAX_MASK_T": 8,
    "MAX_MASK_F": 2,
    "NUM_MASKS": 2,

    # --------------------------------------------------------
    # AMP 混合精度
    # --------------------------------------------------------
    "AMP": True,

    # --------------------------------------------------------
    # ICBHI 评价方式
    #
    # False:
    #   严格四分类。
    #   异常样本只有预测类别完全正确才计入 SE。
    #
    # True:
    #   二分类式评价。
    #   只要异常样本被预测为任意异常类别就计入 SE。
    # --------------------------------------------------------
    "TWO_CLS_EVAL": False,

    # 是否使用类别加权交叉熵
    "WEIGHTED_LOSS": False,

    # 若真实 Mamba 没有成功导入，是否直接停止
    "REQUIRE_MAMBA": True,
}


# ============================================================
# 2. 导入项目中的模型
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. Patch 级 SpecAugment
#
# 输入：
#   x: [948, 768]
#
# 恢复为：
#   [12, 79, 768]
#
# 频率遮挡作用于 12 个频率 patch
# 时间遮挡作用于 79 个时间 patch
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
            "SpecAugment 输入必须为 [N, D]，"
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
        feature_dim
    ).clone()

    for _ in range(num_masks):

        # ----------------------------------------------------
        # 时间 patch 遮挡
        # ----------------------------------------------------
        t_width = random.randint(
            0,
            min(max_mask_t, time_patches)
        )

        if t_width > 0:
            t_start = random.randint(
                0,
                time_patches - t_width
            )

            x_aug[
                :,
                t_start:t_start + t_width,
                :
            ] = 0

        # ----------------------------------------------------
        # 频率 patch 遮挡
        # ----------------------------------------------------
        f_width = random.randint(
            0,
            min(max_mask_f, freq_patches)
        )

        if f_width > 0:
            f_start = random.randint(
                0,
                freq_patches - f_width
            )

            x_aug[
                f_start:f_start + f_width,
                :,
                :
            ] = 0

    return x_aug.reshape(
        expected_tokens,
        feature_dim
    )


# ============================================================
# 5. 数据集
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
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        required_columns = {
            "tokens_path",
            "label",
        }

        missing_columns = (
            required_columns - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"[Dataset] {csv_path} 缺少列："
                f"{sorted(missing_columns)}。"
                f"当前列为：{self.df.columns.tolist()}"
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
                "发现无效标签。四分类标签必须为 0、1、2、3，"
                f"当前无效标签为：{invalid_labels.tolist()}"
            )

        self.class_counts_4 = np.bincount(
            self.labels,
            minlength=4
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
        token_path = str(row["tokens_path"])

        if not os.path.exists(token_path):
            raise FileNotFoundError(
                f"Token 文件不存在：{token_path}"
            )

        tokens_np = np.load(token_path)

        if tokens_np.ndim != 2:
            raise ValueError(
                f"Token 文件必须是二维数组 [N, D]："
                f"{token_path}，"
                f"当前形状为 {tokens_np.shape}"
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
# 6. Collate
#
# 所有样本形状固定为：
#   [948, 768]
#
# 因此不再进行 padding，也不再生成 mask。
# ============================================================
def collate_fixed(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:

    xs, ys = zip(*batch)

    x_batch = torch.stack(
        xs,
        dim=0
    )

    y_batch = torch.stack(
        ys,
        dim=0
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
        hits[0] / (counts[0] + eps)
    )

    # Sensitivity：三个异常类整体正确率
    abnormal_hits = float(
        hits[1] + hits[2] + hits[3]
    )

    abnormal_counts = float(
        counts[1] + counts[2] + counts[3]
    )

    se = 100.0 * (
        abnormal_hits / (abnormal_counts + eps)
    )

    score = (sp + se) / 2.0

    return float(sp), float(se), float(score)


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

    hits = [0.0, 0.0, 0.0, 0.0]
    counts = [0.0, 0.0, 0.0, 0.0]

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
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        batch_size = y.size(0)

        # ----------------------------------------------------
        # 串联模型
        #
        # x:
        #   [B, 948, 768]
        #
        # feat:
        #   [B, 768]
        #
        # logits:
        #   [B, 4]
        # ----------------------------------------------------
        feat = backbone(x)
        logits = classifier(feat)

        loss = criterion(
            logits,
            y
        )

        loss_sum += float(loss.item())
        sample_count += int(batch_size)

        pred = torch.argmax(
            logits,
            dim=1
        )

        all_true.append(
            y.detach().cpu()
        )

        all_pred.append(
            pred.detach().cpu()
        )

        for i in range(batch_size):

            gt = int(y[i].item())
            pr = int(pred[i].item())

            counts[gt] += 1.0

            if not two_cls_eval:
                # 严格四分类评价
                if pr == gt:
                    hits[gt] += 1.0

            else:
                # 二分类式评价
                if gt == 0 and pr == 0:
                    hits[gt] += 1.0

                elif gt != 0 and pr > 0:
                    hits[gt] += 1.0

    sp, se, score = get_score_from_hits_counts(
        hits,
        counts
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
            y_pred
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
            loss_sum / max(1, sample_count)
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
    scheduler=None,
    class_weights: Optional[torch.Tensor] = None,
) -> float:

    backbone.train()
    classifier.train()

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    number_of_batches = len(loader)

    parameters = (
        list(backbone.parameters())
        + list(classifier.parameters())
    )

    for batch_index, (x, y) in enumerate(loader):

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            feat = backbone(x)
            logits = classifier(feat)

            raw_loss = criterion(
                logits,
                y
            )

            loss = (
                raw_loss / accum_steps
            )

        scaler.scale(
            loss
        ).backward()

        running_loss += float(
            raw_loss.detach().item()
        )

        should_step = (
            (batch_index + 1) % accum_steps == 0
            or
            (batch_index + 1) == number_of_batches
        )

        if should_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            if scheduler is not None:
                scheduler.step()

    return (
        running_loss
        / max(1, number_of_batches)
    )


# ============================================================
# 10. AMP Scaler
# ============================================================
def create_grad_scaler(
    use_amp: bool,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=use_amp
        )


# ============================================================
# 11. 模型形状测试
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

    x, y = next(
        iter(loader)
    )

    # 形状测试只取一个样本，减少显存消耗
    x = x[:1].to(device)

    feature = backbone(x)
    logits = classifier(feature)

    print(
        "[SHAPE TEST] input:",
        tuple(x.shape)
    )

    print(
        "[SHAPE TEST] feature:",
        tuple(feature.shape)
    )

    print(
        "[SHAPE TEST] logits:",
        tuple(logits.shape)
    )

    expected_feature_shape = (
        1,
        CONFIG["TOKEN_DIM"],
    )

    expected_logits_shape = (
        1,
        4,
    )

    if tuple(feature.shape) != expected_feature_shape:
        raise RuntimeError(
            "Backbone 输出形状错误："
            f"当前为 {tuple(feature.shape)}，"
            f"要求为 {expected_feature_shape}"
        )

    if tuple(logits.shape) != expected_logits_shape:
        raise RuntimeError(
            "Classifier 输出形状错误："
            f"当前为 {tuple(logits.shape)}，"
            f"要求为 {expected_logits_shape}"
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
        "cuda" if use_cuda else "cpu"
    )

    print(
        f"[INFO] device: {device}"
    )

    if device.type == "cuda":
        print(
            "[INFO] CUDA device count:",
            torch.cuda.device_count()
        )

        print(
            "[INFO] CUDA current device:",
            torch.cuda.current_device()
        )

        print(
            "[INFO] CUDA device name:",
            torch.cuda.get_device_name(
                torch.cuda.current_device()
            )
        )

    # --------------------------------------------------------
    # 检查 Mamba
    # --------------------------------------------------------
    print(
        f"[INFO] HAS_MAMBA = {HAS_MAMBA}"
    )

    if (
        cfg["REQUIRE_MAMBA"]
        and not HAS_MAMBA
    ):
        raise RuntimeError(
            "mamba_ssm 没有成功导入。"
            "当前 model.py 会退化为 GRU，"
            "为防止误训练，程序已经停止。\n"
            "请先运行：\n"
            "python -c \"from mamba_ssm import Mamba; "
            "print('Mamba import success')\""
        )

    # --------------------------------------------------------
    # CSV 路径
    # --------------------------------------------------------
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
            f"找不到训练 CSV：{train_csv}"
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            f"找不到测试 CSV：{test_csv}"
        )

    # --------------------------------------------------------
    # 保存路径
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
        csv_path=str(train_csv),
        is_train=True,
        specaug=bool(cfg["SPEC_AUG"]),
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
        csv_path=str(test_csv),
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

    # Backbone 输出 [B, TOKEN_DIM]
    classifier = nn.Linear(
        int(cfg["TOKEN_DIM"]),
        4,
    ).to(device)

    print(
        "[MODEL] Serial structure:"
    )
    print(
        "[MODEL] Time-Mamba "
        "-> Frequency-Attention "
        "-> Mean Pooling "
        "-> Linear Classifier"
    )

    print(
        "[MODEL] Input shape: "
        f"[B, "
        f"{cfg['FREQ_PATCHES'] * cfg['TIME_PATCHES']}, "
        f"{cfg['TOKEN_DIM']}]"
    )

    # --------------------------------------------------------
    # 形状测试
    # --------------------------------------------------------
    run_shape_test(
        loader=train_loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
    )

    # --------------------------------------------------------
    # 类别权重
    # --------------------------------------------------------
    class_weights = None

    if cfg["WEIGHTED_LOSS"]:

        class_counts = (
            train_dataset
            .class_counts_4
            .astype(np.float32)
        )

        weights = 1.0 / np.maximum(
            class_counts,
            1.0
        )

        weights = (
            weights
            / weights.sum()
            * 4.0
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
            class_counts.tolist()
        )

        print(
            "[INFO] Class weights:",
            weights.tolist()
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
        lr=float(cfg["LR"]),
        weight_decay=float(
            cfg["WEIGHT_DECAY"]
        ),
        betas=(0.9, 0.999),
    )

    # --------------------------------------------------------
    # Scheduler
    #
    # 按实际 optimizer.step() 数量计算
    # --------------------------------------------------------
    accumulation_steps = int(
        cfg["ACCUM_STEPS"]
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader)
        / accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * int(cfg["EPOCHS"])
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                total_optimizer_steps
            ),
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
        f"[INFO] AMP enabled: {use_amp}"
    )

    # --------------------------------------------------------
    # 训练状态
    # --------------------------------------------------------
    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    print()
    print("=" * 70)
    print(
        "Start training: "
        "Serial Time-Mamba "
        "-> Frequency-Attention"
    )
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Epoch
    # --------------------------------------------------------
    for epoch in range(
        1,
        int(cfg["EPOCHS"]) + 1,
    ):
        start_time = time.time()

        train_loss = train_one_epoch(
            loader=train_loader,
            backbone=backbone,
            classifier=classifier,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            accum_steps=accumulation_steps,
            scheduler=scheduler,
            class_weights=class_weights,
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

            marker = "BEST"

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
            time.time() - start_time
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

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
            f"{elapsed_time:.1f}s"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------
        if bad_epochs >= int(
            cfg["PATIENCE"]
        ):
            print(
                "[EARLY STOP] "
                f"连续 {cfg['PATIENCE']} 个 Epoch "
                "Score 没有提升。"
            )

            print(
                f"[EARLY STOP] best epoch = "
                f"{best_epoch}, "
                f"best score = "
                f"{best_score:.4f}"
            )

            break

    # --------------------------------------------------------
    # 最佳结果
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print(
        f"Training completed. "
        f"Best Score = {best_score:.4f}, "
        f"Best Epoch = {best_epoch}"
    )
    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 加载最佳模型并最终评价
    # --------------------------------------------------------
    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    backbone.load_state_dict(
        checkpoint["backbone_state"]
    )

    classifier.load_state_dict(
        checkpoint["classifier_state"]
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
    print("[FINAL RESULT]")
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