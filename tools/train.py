#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

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
)


# ============================================================
# 1. 配置
# ============================================================
CONFIG = {
    # 数据目录
    "ROOT": (
        "/data/dingcong/hybrid/"
        "icbhi_official_ast_patch_tokens"
    ),

    # --------------------------------------------------------
    # 训练参数
    # --------------------------------------------------------
    "EPOCHS": 40,
    "BATCH_SIZE": 4,
    "ACCUM_STEPS": 4,

    "LR": 1e-5,
    "WEIGHT_DECAY": 1e-2,

    "NUM_WORKERS": 1,
    "SEED": 42,
    "DEVICE": "cuda",

    "AMP": True,
    "REQUIRE_MAMBA": True,

    # Early Stopping 只根据严格四分类 Score
    "PATIENCE": 15,

    # 保存目录
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_hierarchical_dual_score"
    ),

    # --------------------------------------------------------
    # Backbone 参数
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
    # 分类头参数
    # --------------------------------------------------------
    "HEAD_DROPOUT": 0.20,

    # 总损失：
    # binary_loss + ABNORMAL_LOSS_WEIGHT * abnormal_loss
    "ABNORMAL_LOSS_WEIGHT": 1.0,

    # 异常三分类权重：
    # weight = 1 / count^power
    "ABNORMAL_WEIGHT_POWER": 0.5,

    "LABEL_SMOOTHING": 0.0,

    # --------------------------------------------------------
    # Binary Head 阈值搜索
    #
    # P(Abnormal) >= threshold：
    #     判定为异常，再由异常三分类头预测 1/2/3
    #
    # P(Abnormal) < threshold：
    #     判定为 Normal
    # --------------------------------------------------------
    "THRESHOLD_MIN": 0.05,
    "THRESHOLD_MAX": 0.95,
    "THRESHOLD_STEP": 0.01,
}


# ============================================================
# 2. 导入项目模型
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


# ============================================================
# 4. Dataset
# ============================================================
class TokenNPY4ClsDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        freq_patches: int,
        time_patches: int,
        input_dim: int,
    ) -> None:
        super().__init__()

        self.csv_path = Path(csv_path)

        self.df = pd.read_csv(
            self.csv_path
        ).reset_index(drop=True)

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
                f"{self.df.columns.tolist()}"
            )

        self.freq_patches = freq_patches
        self.time_patches = time_patches
        self.input_dim = input_dim

        self.expected_tokens = (
            freq_patches * time_patches
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
                "标签必须为 0、1、2、3；"
                f"发现无效标签："
                f"{invalid_labels.tolist()}"
            )

        self.class_counts = np.bincount(
            self.labels,
            minlength=4,
        )

        print(
            f"[Dataset] Loaded {len(self.df)} samples "
            f"from {csv_path}"
        )

        print(
            "[Dataset] class counts:",
            self.class_counts.tolist(),
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(
        self,
        raw_path: str,
    ) -> Path:

        token_path = Path(raw_path)

        if token_path.exists():
            return token_path

        relative_path = (
            self.csv_path.parent
            / token_path
        )

        if relative_path.exists():
            return relative_path

        raise FileNotFoundError(
            f"Token 文件不存在：{raw_path}"
        )

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        row = self.df.iloc[index]

        token_path = self._resolve_path(
            str(row["tokens_path"])
        )

        tokens_np = np.load(
            token_path
        )

        expected_shape = (
            self.expected_tokens,
            self.input_dim,
        )

        if tuple(tokens_np.shape) != expected_shape:
            raise ValueError(
                f"Token shape error：{token_path}\n"
                f"当前形状：{tuple(tokens_np.shape)}\n"
                f"要求形状：{expected_shape}"
            )

        x = torch.from_numpy(
            tokens_np
        ).float()

        y = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long,
        )

        return x, y


# ============================================================
# 5. Collate
# ============================================================
def collate_fixed(
    batch: List[
        Tuple[
            torch.Tensor,
            torch.Tensor,
        ]
    ],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:

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
# 6. 构建异常三分类权重
#
# 类别：
# 1 = Crackle
# 2 = Wheeze
# 3 = Both
# ============================================================
def build_abnormal_class_weights(
    class_counts: np.ndarray,
    power: float,
) -> torch.Tensor:

    abnormal_counts = (
        class_counts[1:4]
        .astype(np.float64)
    )

    weights = (
        1.0
        / np.power(
            np.maximum(
                abnormal_counts,
                1.0,
            ),
            power,
        )
    )

    # 平均权重归一化为 1
    weights = (
        weights
        / weights.mean()
    )

    print(
        "[Loss] Abnormal class counts:",
        abnormal_counts.tolist(),
    )

    print(
        "[Loss] Abnormal class weights:",
        weights.tolist(),
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# 7. 分层预测
# ============================================================
def hierarchical_predict(
    abnormal_probability: np.ndarray,
    abnormal_class_prediction: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    abnormal_probability:
        Binary Head 的 P(Abnormal)，形状 [N]

    abnormal_class_prediction:
        Abnormal Head 输出的最终异常类别，取值为 1、2、3

    threshold:
        当 P(Abnormal) >= threshold 时预测异常；
        否则预测 Normal。
    """

    prediction = np.where(
        abnormal_probability >= threshold,
        abnormal_class_prediction,
        0,
    )

    return prediction.astype(
        np.int64
    )


# ============================================================
# 8. 同时计算二分类和严格四分类指标
# ============================================================
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:

    # ========================================================
    # A. 严格四分类
    #
    # 0 = Normal
    # 1 = Crackle
    # 2 = Wheeze
    # 3 = Both
    # ========================================================
    four_cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
    )

    normal_total = float(
        four_cm[0, :].sum()
    )

    abnormal_total = float(
        four_cm[1:, :].sum()
    )

    # 四分类 SP：
    # Normal 必须预测为 Normal
    four_sp = (
        100.0
        * float(four_cm[0, 0])
        / max(normal_total, 1.0)
    )

    # 四分类 SE：
    # 每个异常类别必须预测为正确的具体类别
    four_abnormal_correct = float(
        four_cm[1, 1]
        + four_cm[2, 2]
        + four_cm[3, 3]
    )

    four_se = (
        100.0
        * four_abnormal_correct
        / max(abnormal_total, 1.0)
    )

    four_score = (
        four_sp + four_se
    ) / 2.0

    # ========================================================
    # B. Normal / Abnormal 二分类
    #
    # 0       -> Normal
    # 1/2/3   -> Abnormal
    # ========================================================
    y_true_binary = (
        y_true > 0
    ).astype(np.int64)

    y_pred_binary = (
        y_pred > 0
    ).astype(np.int64)

    binary_cm = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
    )

    binary_normal_total = float(
        binary_cm[0, :].sum()
    )

    binary_abnormal_total = float(
        binary_cm[1, :].sum()
    )

    # 二分类 SP：
    # Normal 正确识别为 Normal
    binary_sp = (
        100.0
        * float(binary_cm[0, 0])
        / max(binary_normal_total, 1.0)
    )

    # 二分类 SE：
    # 任意异常类别预测为任意异常类别
    binary_se = (
        100.0
        * float(binary_cm[1, 1])
        / max(binary_abnormal_total, 1.0)
    )

    binary_score = (
        binary_sp + binary_se
    ) / 2.0

    # ========================================================
    # C. 辅助指标
    # ========================================================
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
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    predicted_counts = np.bincount(
        y_pred,
        minlength=4,
    )

    return {
        # 严格四分类指标
        "FOUR_SP": float(four_sp),
        "FOUR_SE": float(four_se),
        "FOUR_SCORE": float(four_score),

        # 二分类指标
        "BINARY_SP": float(binary_sp),
        "BINARY_SE": float(binary_se),
        "BINARY_SCORE": float(binary_score),

        # 兼容原来的变量名
        # ICBHI 表示用于选择最佳模型的严格四分类 Score
        "SP": float(four_sp),
        "SE": float(four_se),
        "ICBHI": float(four_score),

        # 辅助指标
        "ACC": float(accuracy),
        "F1": float(macro_f1),

        "class_recall": class_recall,
        "pred_counts": predicted_counts,

        "cm": four_cm,
        "binary_cm": binary_cm,
    }


# ============================================================
# 9. 搜索最佳 Binary 阈值
#
# 最佳阈值只按照严格四分类 Score 选择
# ============================================================
def search_best_threshold(
    abnormal_probability: np.ndarray,
    abnormal_class_prediction: np.ndarray,
    y_true: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> Dict[str, object]:

    thresholds = np.arange(
        threshold_min,
        threshold_max
        + threshold_step / 2.0,
        threshold_step,
    )

    best_metrics = None
    best_threshold = None

    best_balance_difference = float(
        "inf"
    )

    for threshold in thresholds:

        y_pred = hierarchical_predict(
            abnormal_probability=(
                abnormal_probability
            ),

            abnormal_class_prediction=(
                abnormal_class_prediction
            ),

            threshold=float(
                threshold
            ),
        )

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        # 只根据严格四分类 Score 选阈值
        current_score = float(
            metrics["FOUR_SCORE"]
        )

        balance_difference = abs(
            float(metrics["FOUR_SP"])
            - float(metrics["FOUR_SE"])
        )

        if best_metrics is None:
            improved = True

        else:
            best_score = float(
                best_metrics["FOUR_SCORE"]
            )

            improved = (
                current_score
                > best_score + 1e-12
            )

            # Score 完全相同时，
            # 选择 SP 和 SE 更接近的阈值
            if (
                not improved
                and abs(
                    current_score
                    - best_score
                ) <= 1e-12
                and balance_difference
                < best_balance_difference
            ):
                improved = True

        if improved:

            best_metrics = metrics

            best_threshold = float(
                threshold
            )

            best_balance_difference = (
                balance_difference
            )

    if best_metrics is None:
        raise RuntimeError(
            "阈值搜索失败，没有产生有效结果。"
        )

    best_metrics["threshold"] = (
        best_threshold
    )

    return best_metrics


# ============================================================
# 10. 收集模型输出
# ============================================================
@torch.no_grad()
def collect_outputs(
    loader: DataLoader,
    backbone: nn.Module,
    binary_head: nn.Module,
    abnormal_head: nn.Module,
    device: torch.device,
    abnormal_class_weights: torch.Tensor,
    abnormal_loss_weight: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:

    backbone.eval()
    binary_head.eval()
    abnormal_head.eval()

    all_abnormal_probability = []
    all_abnormal_prediction = []
    all_labels = []

    total_binary_loss = 0.0
    total_abnormal_loss = 0.0

    total_samples = 0
    total_abnormal_samples = 0

    binary_criterion = nn.CrossEntropyLoss(
        reduction="sum"
    ).to(device)

    abnormal_criterion = nn.CrossEntropyLoss(
        weight=abnormal_class_weights.to(
            device
        ),
        reduction="sum",
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

        feature = backbone(
            x
        )

        binary_logits = binary_head(
            feature
        )

        abnormal_logits = abnormal_head(
            feature
        )

        # 二分类标签：
        # Normal=0，Abnormal=1
        binary_target = (
            y > 0
        ).long()

        binary_loss = binary_criterion(
            binary_logits,
            binary_target,
        )

        total_binary_loss += float(
            binary_loss.item()
        )

        abnormal_mask = (
            y > 0
        )

        if abnormal_mask.any():

            # 原标签 1/2/3
            # 转成异常头标签 0/1/2
            abnormal_target = (
                y[abnormal_mask] - 1
            )

            abnormal_loss = abnormal_criterion(
                abnormal_logits[
                    abnormal_mask
                ],
                abnormal_target,
            )

            total_abnormal_loss += float(
                abnormal_loss.item()
            )

            total_abnormal_samples += int(
                abnormal_mask.sum().item()
            )

        binary_probability = torch.softmax(
            binary_logits,
            dim=1,
        )

        # P(Abnormal)
        abnormal_probability = (
            binary_probability[:, 1]
        )

        # 异常三分类结果：
        # 0/1/2 -> 最终类别 1/2/3
        abnormal_prediction = (
            torch.argmax(
                abnormal_logits,
                dim=1,
            )
            + 1
        )

        all_abnormal_probability.append(
            abnormal_probability
            .detach()
            .cpu()
        )

        all_abnormal_prediction.append(
            abnormal_prediction
            .detach()
            .cpu()
        )

        all_labels.append(
            y.detach().cpu()
        )

        total_samples += int(
            y.size(0)
        )

    abnormal_probability_np = torch.cat(
        all_abnormal_probability,
        dim=0,
    ).numpy()

    abnormal_prediction_np = torch.cat(
        all_abnormal_prediction,
        dim=0,
    ).numpy()

    labels_np = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    mean_binary_loss = (
        total_binary_loss
        / max(total_samples, 1)
    )

    mean_abnormal_loss = (
        total_abnormal_loss
        / max(
            total_abnormal_samples,
            1,
        )
    )

    mean_total_loss = (
        mean_binary_loss
        + abnormal_loss_weight
        * mean_abnormal_loss
    )

    return (
        abnormal_probability_np,
        abnormal_prediction_np,
        labels_np,

        float(mean_total_loss),
        float(mean_binary_loss),
        float(mean_abnormal_loss),
    )


# ============================================================
# 11. 搜索最佳阈值并评价
# ============================================================
@torch.no_grad()
def evaluate_with_threshold_search(
    loader: DataLoader,
    backbone: nn.Module,
    binary_head: nn.Module,
    abnormal_head: nn.Module,
    device: torch.device,
    abnormal_class_weights: torch.Tensor,
    abnormal_loss_weight: float,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> Dict[str, object]:

    (
        abnormal_probability,
        abnormal_prediction,
        y_true,

        total_loss,
        binary_loss,
        abnormal_loss,
    ) = collect_outputs(
        loader=loader,

        backbone=backbone,
        binary_head=binary_head,
        abnormal_head=abnormal_head,

        device=device,

        abnormal_class_weights=(
            abnormal_class_weights
        ),

        abnormal_loss_weight=(
            abnormal_loss_weight
        ),
    )

    metrics = search_best_threshold(
        abnormal_probability=(
            abnormal_probability
        ),

        abnormal_class_prediction=(
            abnormal_prediction
        ),

        y_true=y_true,

        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )

    metrics["LOSS"] = total_loss
    metrics["BINARY_LOSS"] = binary_loss
    metrics["ABNORMAL_LOSS"] = abnormal_loss

    return metrics


# ============================================================
# 12. 使用固定阈值评价
# ============================================================
@torch.no_grad()
def evaluate_with_fixed_threshold(
    loader: DataLoader,
    backbone: nn.Module,
    binary_head: nn.Module,
    abnormal_head: nn.Module,
    device: torch.device,
    abnormal_class_weights: torch.Tensor,
    abnormal_loss_weight: float,
    threshold: float,
) -> Dict[str, object]:

    (
        abnormal_probability,
        abnormal_prediction,
        y_true,

        total_loss,
        binary_loss,
        abnormal_loss,
    ) = collect_outputs(
        loader=loader,

        backbone=backbone,
        binary_head=binary_head,
        abnormal_head=abnormal_head,

        device=device,

        abnormal_class_weights=(
            abnormal_class_weights
        ),

        abnormal_loss_weight=(
            abnormal_loss_weight
        ),
    )

    y_pred = hierarchical_predict(
        abnormal_probability=(
            abnormal_probability
        ),

        abnormal_class_prediction=(
            abnormal_prediction
        ),

        threshold=threshold,
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    metrics["LOSS"] = total_loss
    metrics["BINARY_LOSS"] = binary_loss
    metrics["ABNORMAL_LOSS"] = abnormal_loss

    metrics["threshold"] = float(
        threshold
    )

    return metrics


# ============================================================
# 13. AMP GradScaler
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
# 14. 训练一个 Epoch
# ============================================================
def train_one_epoch(
    loader: DataLoader,
    backbone: nn.Module,
    binary_head: nn.Module,
    abnormal_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
    accum_steps: int,
    abnormal_class_weights: torch.Tensor,
    abnormal_loss_weight: float,
    label_smoothing: float,
) -> Tuple[
    float,
    float,
    float,
    int,
]:

    backbone.train()
    binary_head.train()
    abnormal_head.train()

    # Normal 与全部 Abnormal 数量近似平衡，
    # 二分类损失不使用类别权重
    binary_criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )

    # 异常三分类使用温和类别权重
    abnormal_criterion = nn.CrossEntropyLoss(
        weight=abnormal_class_weights.to(
            device
        ),
        label_smoothing=label_smoothing,
    )

    parameters = (
        list(backbone.parameters())
        + list(binary_head.parameters())
        + list(abnormal_head.parameters())
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss_sum = 0.0
    binary_loss_sum = 0.0
    abnormal_loss_sum = 0.0

    optimizer_steps = 0
    number_of_batches = len(loader)

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

            binary_logits = binary_head(
                feature
            )

            abnormal_logits = abnormal_head(
                feature
            )

            # 0 = Normal
            # 1 = Abnormal
            binary_target = (
                y > 0
            ).long()

            binary_loss = binary_criterion(
                binary_logits,
                binary_target,
            )

            abnormal_mask = (
                y > 0
            )

            if abnormal_mask.any():

                # 标签 1/2/3 转为 0/1/2
                abnormal_target = (
                    y[abnormal_mask] - 1
                )

                abnormal_loss = abnormal_criterion(
                    abnormal_logits[
                        abnormal_mask
                    ],
                    abnormal_target,
                )

            else:

                abnormal_loss = torch.zeros(
                    (),
                    device=device,
                    dtype=binary_loss.dtype,
                )

            raw_loss = (
                binary_loss
                + abnormal_loss_weight
                * abnormal_loss
            )

            loss = (
                raw_loss
                / accum_steps
            )

        scaler.scale(
            loss
        ).backward()

        total_loss_sum += float(
            raw_loss.detach().item()
        )

        binary_loss_sum += float(
            binary_loss.detach().item()
        )

        abnormal_loss_sum += float(
            abnormal_loss.detach().item()
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

            # Scale 没有下降，说明参数更新没有被跳过
            if new_scale >= old_scale:
                optimizer_steps += 1

            optimizer.zero_grad(
                set_to_none=True
            )

    denominator = max(
        number_of_batches,
        1,
    )

    mean_total_loss = (
        total_loss_sum
        / denominator
    )

    mean_binary_loss = (
        binary_loss_sum
        / denominator
    )

    mean_abnormal_loss = (
        abnormal_loss_sum
        / denominator
    )

    return (
        float(mean_total_loss),
        float(mean_binary_loss),
        float(mean_abnormal_loss),
        optimizer_steps,
    )


# ============================================================
# 15. 模型形状检查
# ============================================================
@torch.no_grad()
def run_shape_test(
    loader: DataLoader,
    backbone: nn.Module,
    binary_head: nn.Module,
    abnormal_head: nn.Module,
    device: torch.device,
    expected_dim: int,
) -> None:

    backbone.eval()
    binary_head.eval()
    abnormal_head.eval()

    x, _ = next(
        iter(loader)
    )

    x = x[:1].to(
        device
    )

    feature = backbone(
        x
    )

    binary_logits = binary_head(
        feature
    )

    abnormal_logits = abnormal_head(
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
        "[SHAPE TEST] binary logits:",
        tuple(binary_logits.shape),
    )

    print(
        "[SHAPE TEST] abnormal logits:",
        tuple(abnormal_logits.shape),
    )

    if tuple(feature.shape) != (
        1,
        expected_dim,
    ):
        raise RuntimeError(
            f"Backbone 输出为 "
            f"{tuple(feature.shape)}，"
            f"要求为 (1, {expected_dim})"
        )

    if tuple(binary_logits.shape) != (
        1,
        2,
    ):
        raise RuntimeError(
            f"Binary Head 输出为 "
            f"{tuple(binary_logits.shape)}，"
            "要求为 (1, 2)"
        )

    if tuple(abnormal_logits.shape) != (
        1,
        3,
    ):
        raise RuntimeError(
            f"Abnormal Head 输出为 "
            f"{tuple(abnormal_logits.shape)}，"
            "要求为 (1, 3)"
        )

    print(
        "[SHAPE TEST] Passed."
    )


# ============================================================
# 16. 转换为可保存类型
# ============================================================
def serializable_metrics(
    metrics: Dict[str, object],
) -> Dict[str, object]:

    result = {}

    for key, value in metrics.items():

        if isinstance(
            value,
            np.ndarray,
        ):
            result[key] = value.tolist()

        else:
            result[key] = value

    return result


# ============================================================
# 17. 保存 Checkpoint
# ============================================================
def save_checkpoint(
    path: str,
    epoch: int,
    backbone: nn.Module,
    binary_head: nn.Module,
    abnormal_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: Dict[str, object],
    abnormal_class_weights: torch.Tensor,
    config: Dict[str, object],
) -> None:

    torch.save(
        {
            "epoch": epoch,

            "backbone_state": (
                backbone.state_dict()
            ),

            "binary_head_state": (
                binary_head.state_dict()
            ),

            "abnormal_head_state": (
                abnormal_head.state_dict()
            ),

            "optimizer_state": (
                optimizer.state_dict()
            ),

            "scheduler_state": (
                scheduler.state_dict()
            ),

            "metrics": (
                serializable_metrics(
                    metrics
                )
            ),

            # 用于选模型的严格四分类 Score
            "score": float(
                metrics["FOUR_SCORE"]
            ),

            "four_score": float(
                metrics["FOUR_SCORE"]
            ),

            "binary_score": float(
                metrics["BINARY_SCORE"]
            ),

            "threshold": float(
                metrics["threshold"]
            ),

            "abnormal_class_weights": (
                abnormal_class_weights
                .detach()
                .cpu()
                .tolist()
            ),

            "config": deepcopy(
                config
            ),
        },
        path,
    )


# ============================================================
# 18. 打印最终结果
# ============================================================
def print_metrics(
    title: str,
    metrics: Dict[str, object],
) -> None:

    print()
    print("=" * 76)
    print(f"[{title}]")
    print("=" * 76)

    print(
        f"Threshold: "
        f"{metrics['threshold']:.4f}"
    )

    print()
    print(
        "----- Strict Four-Class Evaluation -----"
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

    print()
    print(
        "----- Normal / Abnormal Binary Evaluation -----"
    )

    print(
        f"Binary Score: "
        f"{metrics['BINARY_SCORE']:.4f}"
    )

    print(
        f"Binary SP: "
        f"{metrics['BINARY_SP']:.4f}"
    )

    print(
        f"Binary SE: "
        f"{metrics['BINARY_SE']:.4f}"
    )

    print()
    print(
        "----- Auxiliary Metrics -----"
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
        f"Total loss: "
        f"{metrics['LOSS']:.4f}"
    )

    print(
        f"Binary loss: "
        f"{metrics['BINARY_LOSS']:.4f}"
    )

    print(
        f"Abnormal loss: "
        f"{metrics['ABNORMAL_LOSS']:.4f}"
    )

    print(
        "Per-class recall:",
        np.round(
            metrics["class_recall"],
            4,
        ).tolist(),
    )

    print(
        "Predicted counts:",
        metrics[
            "pred_counts"
        ].tolist(),
    )

    print()
    print(
        "Four-Class Confusion Matrix:"
    )

    print(
        "Rows/Columns = "
        "Normal, Crackle, Wheeze, Both"
    )

    print(
        metrics["cm"]
    )

    print()
    print(
        "Binary Confusion Matrix:"
    )

    print(
        "Rows/Columns = Normal, Abnormal"
    )

    print(
        metrics["binary_cm"]
    )


# ============================================================
# 19. 主函数
# ============================================================
def main() -> None:

    cfg = CONFIG

    set_seed(
        int(cfg["SEED"])
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device(
        "cuda"
        if (
            cfg["DEVICE"] == "cuda"
            and torch.cuda.is_available()
        )
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
            "已停止训练。"
        )

    # --------------------------------------------------------
    # 数据路径
    # --------------------------------------------------------
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
            f"找不到：{train_csv}"
        )

    if not test_csv.exists():
        raise FileNotFoundError(
            f"找不到：{test_csv}"
        )

    # 优先使用独立验证集选择 Epoch 和阈值
    if val_csv.exists():

        selection_csv = val_csv

        print(
            "[INFO] 使用 val_index.csv "
            "选择最佳 Epoch 和阈值：",
            selection_csv,
        )

    else:

        selection_csv = test_csv

        print(
            "[WARNING] 未找到 val_index.csv。"
        )

        print(
            "[WARNING] 当前使用 test_index.csv "
            "选择最佳 Epoch 和阈值。"
        )

        print(
            "[WARNING] 正式论文实验应使用独立验证集，"
            "官方测试集只评价一次。"
        )

    # --------------------------------------------------------
    # 保存路径
    # --------------------------------------------------------
    os.makedirs(
        cfg["SAVE_DIR"],
        exist_ok=True,
    )

    best_score_path = os.path.join(
        cfg["SAVE_DIR"],
        "best_four_class_score.pth",
    )

    last_path = os.path.join(
        cfg["SAVE_DIR"],
        "last.pth",
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset_args = {
        "freq_patches": int(
            cfg["FREQ_PATCHES"]
        ),

        "time_patches": int(
            cfg["TIME_PATCHES"]
        ),

        "input_dim": int(
            cfg["INPUT_DIM"]
        ),
    }

    train_dataset = TokenNPY4ClsDataset(
        csv_path=str(train_csv),
        **dataset_args,
    )

    selection_dataset = TokenNPY4ClsDataset(
        csv_path=str(selection_csv),
        **dataset_args,
    )

    test_dataset = TokenNPY4ClsDataset(
        csv_path=str(test_csv),
        **dataset_args,
    )

    # --------------------------------------------------------
    # 异常三分类权重
    # --------------------------------------------------------
    abnormal_class_weights = (
        build_abnormal_class_weights(
            class_counts=(
                train_dataset.class_counts
            ),

            power=float(
                cfg[
                    "ABNORMAL_WEIGHT_POWER"
                ]
            ),
        )
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    loader_args = {
        "batch_size": int(
            cfg["BATCH_SIZE"]
        ),

        "num_workers": int(
            cfg["NUM_WORKERS"]
        ),

        "pin_memory": (
            device.type == "cuda"
        ),

        "collate_fn": collate_fixed,

        "drop_last": False,

        "persistent_workers": (
            int(
                cfg["NUM_WORKERS"]
            )
            > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_args,
    )

    selection_loader = DataLoader(
        selection_dataset,
        shuffle=False,
        **loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_args,
    )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------
    backbone = TimeFrequencyEncoder(
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
    ).to(device)

    # --------------------------------------------------------
    # Binary Head
    #
    # 0 = Normal
    # 1 = Abnormal
    # --------------------------------------------------------
    binary_head = nn.Sequential(
        nn.Dropout(
            float(
                cfg["HEAD_DROPOUT"]
            )
        ),

        nn.Linear(
            int(cfg["D_MODEL"]),
            2,
        ),
    ).to(device)

    # --------------------------------------------------------
    # Abnormal Head
    #
    # 0 = Crackle -> 最终类别 1
    # 1 = Wheeze  -> 最终类别 2
    # 2 = Both    -> 最终类别 3
    # --------------------------------------------------------
    abnormal_head = nn.Sequential(
        nn.Dropout(
            float(
                cfg["HEAD_DROPOUT"]
            )
        ),

        nn.Linear(
            int(cfg["D_MODEL"]),
            3,
        ),
    ).to(device)

    print(
        "[MODEL] Hierarchical classification"
    )

    print(
        "[MODEL] Binary Head: "
        "Normal / Abnormal"
    )

    print(
        "[MODEL] Abnormal Head: "
        "Crackle / Wheeze / Both"
    )

    print(
        "[MODEL] Best checkpoint selected by "
        "strict 4-Class Score"
    )

    # --------------------------------------------------------
    # Shape Test
    # --------------------------------------------------------
    run_shape_test(
        loader=train_loader,

        backbone=backbone,
        binary_head=binary_head,
        abnormal_head=abnormal_head,

        device=device,

        expected_dim=int(
            cfg["D_MODEL"]
        ),
    )

    parameters = (
        list(backbone.parameters())
        + list(binary_head.parameters())
        + list(abnormal_head.parameters())
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

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        parameters,

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

    # 每个 Epoch 更新一次
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
    # 最佳结果
    #
    # 只使用严格四分类 Score 选模型
    # --------------------------------------------------------
    best_four_score = -1.0
    best_binary_score = -1.0

    best_score_epoch = -1
    best_threshold = 0.5

    bad_epochs = 0

    print()
    print(
        "=" * 100
    )

    print(
        "Start training: "
        "Hierarchical Classification "
        "+ Binary and Four-Class Scores"
    )

    print(
        "=" * 100
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

        (
            train_total_loss,
            train_binary_loss,
            train_abnormal_loss,
            optimizer_steps,
        ) = train_one_epoch(
            loader=train_loader,

            backbone=backbone,
            binary_head=binary_head,
            abnormal_head=abnormal_head,

            optimizer=optimizer,
            device=device,

            use_amp=use_amp,
            scaler=scaler,

            accum_steps=int(
                cfg["ACCUM_STEPS"]
            ),

            abnormal_class_weights=(
                abnormal_class_weights
            ),

            abnormal_loss_weight=float(
                cfg[
                    "ABNORMAL_LOSS_WEIGHT"
                ]
            ),

            label_smoothing=float(
                cfg["LABEL_SMOOTHING"]
            ),
        )

        metrics = evaluate_with_threshold_search(
            loader=selection_loader,

            backbone=backbone,
            binary_head=binary_head,
            abnormal_head=abnormal_head,

            device=device,

            abnormal_class_weights=(
                abnormal_class_weights
            ),

            abnormal_loss_weight=float(
                cfg[
                    "ABNORMAL_LOSS_WEIGHT"
                ]
            ),

            threshold_min=float(
                cfg["THRESHOLD_MIN"]
            ),

            threshold_max=float(
                cfg["THRESHOLD_MAX"]
            ),

            threshold_step=float(
                cfg["THRESHOLD_STEP"]
            ),
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        if optimizer_steps > 0:
            scheduler.step()

        four_score = float(
            metrics["FOUR_SCORE"]
        )

        binary_score = float(
            metrics["BINARY_SCORE"]
        )

        threshold = float(
            metrics["threshold"]
        )

        # 仅作为记录，不用于选模型
        best_binary_score = max(
            best_binary_score,
            binary_score,
        )

        # ----------------------------------------------------
        # 最佳模型只根据严格四分类 Score
        # ----------------------------------------------------
        if (
            four_score
            > best_four_score + 1e-9
        ):

            best_four_score = four_score
            best_score_epoch = epoch
            best_threshold = threshold

            bad_epochs = 0
            marker = "BEST-4SCORE"

            save_checkpoint(
                path=best_score_path,
                epoch=epoch,

                backbone=backbone,
                binary_head=binary_head,
                abnormal_head=abnormal_head,

                optimizer=optimizer,
                scheduler=scheduler,

                metrics=metrics,

                abnormal_class_weights=(
                    abnormal_class_weights
                ),

                config=cfg,
            )

        else:

            bad_epochs += 1
            marker = "-"

        # 保存最后一轮
        save_checkpoint(
            path=last_path,
            epoch=epoch,

            backbone=backbone,
            binary_head=binary_head,
            abnormal_head=abnormal_head,

            optimizer=optimizer,
            scheduler=scheduler,

            metrics=metrics,

            abnormal_class_weights=(
                abnormal_class_weights
            ),

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

        elapsed_time = (
            time.time()
            - start_time
        )

        print(
            f"[{marker}] "
            f"Epoch {epoch:03d}/"
            f"{cfg['EPOCHS']} | "

            f"train "
            f"{train_total_loss:.4f} | "

            f"bin_loss "
            f"{train_binary_loss:.4f} | "

            f"abn_loss "
            f"{train_abnormal_loss:.4f} | "

            f"4-Score "
            f"{metrics['FOUR_SCORE']:.4f} | "

            f"4-SP "
            f"{metrics['FOUR_SP']:.4f} | "

            f"4-SE "
            f"{metrics['FOUR_SE']:.4f} | "

            f"Binary-Score "
            f"{metrics['BINARY_SCORE']:.4f} | "

            f"Binary-SP "
            f"{metrics['BINARY_SP']:.4f} | "

            f"Binary-SE "
            f"{metrics['BINARY_SE']:.4f} | "

            f"Thr "
            f"{threshold:.2f} | "

            f"ACC "
            f"{metrics['ACC']:.4f} | "

            f"F1 "
            f"{metrics['F1']:.4f} | "

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

        # ----------------------------------------------------
        # Early Stopping 只根据严格四分类 Score
        # ----------------------------------------------------
        if (
            bad_epochs
            >= int(
                cfg["PATIENCE"]
            )
        ):

            print(
                "[EARLY STOP] "
                f"4-Class Score 连续 "
                f"{cfg['PATIENCE']} "
                "个 Epoch 没有提升。"
            )

            print(
                f"[EARLY STOP] "
                f"Best 4-Class Score = "
                f"{best_four_score:.4f}, "
                f"Epoch = "
                f"{best_score_epoch}, "
                f"Threshold = "
                f"{best_threshold:.4f}"
            )

            break

    # --------------------------------------------------------
    # 加载最佳四分类 Score 模型
    # --------------------------------------------------------
    checkpoint = torch.load(
        best_score_path,
        map_location=device,
    )

    backbone.load_state_dict(
        checkpoint[
            "backbone_state"
        ]
    )

    binary_head.load_state_dict(
        checkpoint[
            "binary_head_state"
        ]
    )

    abnormal_head.load_state_dict(
        checkpoint[
            "abnormal_head_state"
        ]
    )

    best_threshold = float(
        checkpoint["threshold"]
    )

    print()
    print(
        "=" * 100
    )

    print(
        f"Best 4-Class Score: "
        f"{best_four_score:.4f}"
    )

    print(
        f"Best observed Binary Score: "
        f"{best_binary_score:.4f}"
    )

    print(
        f"Best Epoch: "
        f"{best_score_epoch}"
    )

    print(
        f"Best Threshold: "
        f"{best_threshold:.4f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_score_path}"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # 最终测试
    #
    # 使用最佳四分类模型和对应固定阈值
    # --------------------------------------------------------
    final_metrics = evaluate_with_fixed_threshold(
        loader=test_loader,

        backbone=backbone,
        binary_head=binary_head,
        abnormal_head=abnormal_head,

        device=device,

        abnormal_class_weights=(
            abnormal_class_weights
        ),

        abnormal_loss_weight=float(
            cfg[
                "ABNORMAL_LOSS_WEIGHT"
            ]
        ),

        threshold=best_threshold,
    )

    print_metrics(
        title="FINAL TEST RESULT",
        metrics=final_metrics,
    )


if __name__ == "__main__":
    main()