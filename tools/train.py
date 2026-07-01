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

    # 只根据 Score 执行 Early Stopping
    "PATIENCE": 15,

    # 新目录，避免覆盖之前模型
    "SAVE_DIR": (
        "/data/dingcong/hybrid/"
        "checkpoints_serial_256_no_sampler_threshold"
    ),

    # --------------------------------------------------------
    # 模型参数
    # --------------------------------------------------------
    "INPUT_DIM": 768,
    "D_MODEL": 256,

    "FREQ_PATCHES": 12,
    "TIME_PATCHES": 79,

    "TIME_DEPTH": 1,
    "FREQ_DEPTH": 1,

    "NHEAD": 8,

    "DROPOUT": 0.15,
    "CLASSIFIER_DROPOUT": 0.20,

    # --------------------------------------------------------
    # 损失函数
    #
    # 不使用 WeightedRandomSampler
    # 不使用类别权重
    # 不使用 Label Smoothing
    # --------------------------------------------------------
    "LABEL_SMOOTHING": 0.0,

    # --------------------------------------------------------
    # Normal 阈值搜索
    #
    # 若 P(Normal) >= threshold，则预测 Normal；
    # 否则在 Crackle、Wheeze、Both 中选择最大概率类别。
    # --------------------------------------------------------
    "SEARCH_THRESHOLD": True,

    "THRESHOLD_MIN": 0.20,
    "THRESHOLD_MAX": 0.80,
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
                f"当前列：{self.df.columns.tolist()}"
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
# 5. 固定形状 Collate
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
# 6. 根据 Normal 阈值生成四分类预测
# ============================================================
def predict_with_normal_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    probabilities:
        [N, 4]

    规则：
        P(Normal) >= threshold
            -> 类别 0

        P(Normal) < threshold
            -> 在类别 1、2、3 中取最大概率类别
    """

    normal_probability = probabilities[:, 0]

    abnormal_prediction = (
        np.argmax(
            probabilities[:, 1:],
            axis=1,
        )
        + 1
    )

    prediction = np.where(
        normal_probability >= threshold,
        0,
        abnormal_prediction,
    )

    return prediction.astype(
        np.int64
    )


# ============================================================
# 7. 计算严格四分类 Score
#
# SP：
#   Normal 类正确率
#
# SE：
#   Crackle、Wheeze、Both 三个异常类别中，
#   具体类别预测正确的总体比例
#
# Score：
#   (SP + SE) / 2
# ============================================================
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
    )

    normal_total = float(
        cm[0, :].sum()
    )

    abnormal_total = float(
        cm[1:, :].sum()
    )

    normal_correct = float(
        cm[0, 0]
    )

    abnormal_correct = float(
        cm[1, 1]
        + cm[2, 2]
        + cm[3, 3]
    )

    sp = (
        100.0
        * normal_correct
        / max(normal_total, 1.0)
    )

    se = (
        100.0
        * abnormal_correct
        / max(abnormal_total, 1.0)
    )

    score = (
        sp + se
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
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    predicted_counts = np.bincount(
        y_pred,
        minlength=4,
    )

    return {
        "SP": float(sp),
        "SE": float(se),
        "ICBHI": float(score),
        "ACC": float(accuracy),
        "F1": float(macro_f1),

        "class_recall": class_recall,
        "pred_counts": predicted_counts,
        "cm": cm,
    }


# ============================================================
# 8. 搜索最佳 Normal 阈值
# ============================================================
def search_best_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> Dict[str, object]:

    thresholds = np.arange(
        threshold_min,
        threshold_max + threshold_step / 2.0,
        threshold_step,
    )

    best_metrics = None
    best_threshold = None
    best_balance_difference = float("inf")

    for threshold in thresholds:

        y_pred = predict_with_normal_threshold(
            probabilities=probabilities,
            threshold=float(threshold),
        )

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        score = float(
            metrics["ICBHI"]
        )

        balance_difference = abs(
            float(metrics["SP"])
            - float(metrics["SE"])
        )

        if best_metrics is None:
            improved = True

        else:
            best_score = float(
                best_metrics["ICBHI"]
            )

            improved = (
                score > best_score + 1e-12
            )

            # Score 相同时，选择 SP 与 SE 更接近的阈值
            if (
                not improved
                and abs(score - best_score) <= 1e-12
                and balance_difference
                < best_balance_difference
            ):
                improved = True

        if improved:
            best_metrics = metrics
            best_threshold = float(threshold)

            best_balance_difference = (
                balance_difference
            )

    best_metrics["threshold"] = (
        best_threshold
    )

    return best_metrics


# ============================================================
# 9. 收集验证集概率
# ============================================================
@torch.no_grad()
def collect_probabilities(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    device: torch.device,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
]:

    backbone.eval()
    classifier.eval()

    all_probabilities = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

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

        logits = classifier(
            feature
        )

        loss = criterion(
            logits,
            y,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        total_loss += float(
            loss.item()
        )

        total_samples += int(
            y.size(0)
        )

        all_probabilities.append(
            probabilities.detach().cpu()
        )

        all_labels.append(
            y.detach().cpu()
        )

    probabilities_np = torch.cat(
        all_probabilities,
        dim=0,
    ).numpy()

    labels_np = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    mean_loss = (
        total_loss
        / max(total_samples, 1)
    )

    return (
        probabilities_np,
        labels_np,
        float(mean_loss),
    )


# ============================================================
# 10. 验证并搜索最佳阈值
# ============================================================
@torch.no_grad()
def evaluate_with_threshold_search(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> Dict[str, object]:

    (
        probabilities,
        y_true,
        mean_loss,
    ) = collect_probabilities(
        loader=loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
    )

    metrics = search_best_threshold(
        probabilities=probabilities,
        y_true=y_true,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )

    metrics["LOSS"] = mean_loss

    return metrics


# ============================================================
# 11. 使用固定阈值评价
# ============================================================
@torch.no_grad()
def evaluate_with_fixed_threshold(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, object]:

    (
        probabilities,
        y_true,
        mean_loss,
    ) = collect_probabilities(
        loader=loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
    )

    y_pred = predict_with_normal_threshold(
        probabilities=probabilities,
        threshold=threshold,
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    metrics["LOSS"] = mean_loss
    metrics["threshold"] = float(
        threshold
    )

    return metrics


# ============================================================
# 12. AMP GradScaler
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
# 13. 训练一个 Epoch
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
    label_smoothing: float,
) -> Tuple[
    float,
    int,
]:

    backbone.train()
    classifier.train()

    # 普通交叉熵：
    # 不使用类别权重
    # 不使用 WeightedRandomSampler
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    parameters = (
        list(backbone.parameters())
        + list(classifier.parameters())
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    optimizer_steps = 0

    number_of_batches = len(
        loader
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

            feature = backbone(
                x
            )

            logits = classifier(
                feature
            )

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

        total_loss += float(
            raw_loss.detach().item()
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

            if new_scale >= old_scale:
                optimizer_steps += 1

            optimizer.zero_grad(
                set_to_none=True
            )

    mean_loss = (
        total_loss
        / max(number_of_batches, 1)
    )

    return (
        mean_loss,
        optimizer_steps,
    )


# ============================================================
# 14. 模型形状检查
# ============================================================
@torch.no_grad()
def run_shape_test(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    expected_dim: int,
) -> None:

    backbone.eval()
    classifier.eval()

    x, _ = next(
        iter(loader)
    )

    x = x[:1].to(
        device
    )

    feature = backbone(
        x
    )

    logits = classifier(
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
        "[SHAPE TEST] logits:",
        tuple(logits.shape),
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

    if tuple(logits.shape) != (
        1,
        4,
    ):
        raise RuntimeError(
            f"Classifier 输出为 "
            f"{tuple(logits.shape)}，"
            "要求为 (1, 4)"
        )

    print(
        "[SHAPE TEST] Passed."
    )


# ============================================================
# 15. 将指标转换为可保存类型
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
            result[key] = (
                value.tolist()
            )
        else:
            result[key] = value

    return result


# ============================================================
# 16. 保存模型
# ============================================================
def save_checkpoint(
    path: str,
    epoch: int,
    backbone: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: Dict[str, object],
    config: Dict[str, object],
) -> None:

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

            "metrics": (
                serializable_metrics(
                    metrics
                )
            ),

            "score": float(
                metrics["ICBHI"]
            ),

            "threshold": float(
                metrics["threshold"]
            ),

            "config": deepcopy(
                config
            ),
        },
        path,
    )


# ============================================================
# 17. 输出结果
# ============================================================
def print_metrics(
    title: str,
    metrics: Dict[str, object],
) -> None:

    print()
    print(
        f"[{title}]"
    )

    print(
        f"Threshold: "
        f"{metrics['threshold']:.4f}"
    )

    print(
        f"Score: "
        f"{metrics['ICBHI']:.4f}"
    )

    print(
        f"SP: "
        f"{metrics['SP']:.4f}"
    )

    print(
        f"SE: "
        f"{metrics['SE']:.4f}"
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

    print(
        "Confusion Matrix:"
    )

    print(
        metrics["cm"]
    )


# ============================================================
# 18. 主函数
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

    if val_csv.exists():

        selection_csv = val_csv

        print(
            "[INFO] 使用 val_index.csv "
            "选择 Epoch 和阈值：",
            selection_csv,
        )

    else:

        selection_csv = test_csv

        print(
            "[WARNING] 未找到 val_index.csv。"
        )

        print(
            "[WARNING] 当前将使用 test_index.csv "
            "搜索阈值和选择模型。"
        )

        print(
            "[WARNING] 正式论文实验建议从官方训练集 "
            "划出独立验证集。"
        )

    # --------------------------------------------------------
    # 保存目录
    # --------------------------------------------------------
    os.makedirs(
        cfg["SAVE_DIR"],
        exist_ok=True,
    )

    best_score_path = os.path.join(
        cfg["SAVE_DIR"],
        "best_score.pth",
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
    # DataLoader
    #
    # 不使用 WeightedRandomSampler
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
    # Classifier
    # --------------------------------------------------------
    classifier = nn.Sequential(
        nn.Dropout(
            float(
                cfg[
                    "CLASSIFIER_DROPOUT"
                ]
            )
        ),

        nn.Linear(
            int(
                cfg["D_MODEL"]
            ),
            4,
        ),
    ).to(device)

    print(
        "[MODEL] 768 -> 256 Projection"
    )

    print(
        "[MODEL] Time-Mamba "
        "-> Frequency-Attention"
    )

    print(
        "[MODEL] No WeightedRandomSampler"
    )

    print(
        "[MODEL] Normal threshold search enabled"
    )

    # --------------------------------------------------------
    # Shape Test
    # --------------------------------------------------------
    run_shape_test(
        loader=train_loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
        expected_dim=int(
            cfg["D_MODEL"]
        ),
    )

    parameters = (
        list(backbone.parameters())
        + list(classifier.parameters())
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
    # --------------------------------------------------------
    best_score = -1.0
    best_score_epoch = -1
    best_threshold = 0.5

    bad_epochs = 0

    print()
    print(
        "=" * 76
    )

    print(
        "Start training: "
        "No Sampler + Normal Threshold Search"
    )

    print(
        "=" * 76
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

        train_loss, optimizer_steps = (
            train_one_epoch(
                loader=train_loader,
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                device=device,
                use_amp=use_amp,
                scaler=scaler,

                accum_steps=int(
                    cfg[
                        "ACCUM_STEPS"
                    ]
                ),

                label_smoothing=float(
                    cfg[
                        "LABEL_SMOOTHING"
                    ]
                ),
            )
        )

        # 每一轮在验证集上搜索最佳 Normal 阈值
        metrics = evaluate_with_threshold_search(
            loader=selection_loader,
            backbone=backbone,
            classifier=classifier,
            device=device,

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

        score = float(
            metrics["ICBHI"]
        )

        threshold = float(
            metrics["threshold"]
        )

        # ----------------------------------------------------
        # 只根据 Score 保存最佳模型
        # ----------------------------------------------------
        if score > best_score + 1e-9:

            best_score = score
            best_score_epoch = epoch
            best_threshold = threshold

            bad_epochs = 0
            marker = "BEST-SCORE"

            save_checkpoint(
                path=best_score_path,
                epoch=epoch,
                backbone=backbone,
                classifier=classifier,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                config=cfg,
            )

        else:

            bad_epochs += 1
            marker = "-"

        save_checkpoint(
            path=last_path,
            epoch=epoch,
            backbone=backbone,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
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
            f"{train_loss:.4f} | "

            f"val "
            f"{metrics['LOSS']:.4f} | "

            f"Score "
            f"{score:.4f} | "

            f"SP "
            f"{metrics['SP']:.4f} | "

            f"SE "
            f"{metrics['SE']:.4f} | "

            f"Threshold "
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
        # Early Stopping 只看 Score
        # ----------------------------------------------------
        if (
            bad_epochs
            >= int(
                cfg["PATIENCE"]
            )
        ):

            print(
                "[EARLY STOP] "
                f"Score 连续 "
                f"{cfg['PATIENCE']} "
                "个 Epoch 没有提升。"
            )

            print(
                f"[EARLY STOP] "
                f"Best Score = "
                f"{best_score:.4f}, "
                f"Epoch = "
                f"{best_score_epoch}, "
                f"Threshold = "
                f"{best_threshold:.4f}"
            )

            break

    # --------------------------------------------------------
    # 加载最佳模型
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

    classifier.load_state_dict(
        checkpoint[
            "classifier_state"
        ]
    )

    best_threshold = float(
        checkpoint["threshold"]
    )

    print()
    print(
        "=" * 76
    )

    print(
        f"Best Score: "
        f"{best_score:.4f}"
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
        "=" * 76
    )

    # --------------------------------------------------------
    # 使用最佳阈值在官方测试集评价
    # --------------------------------------------------------
    final_metrics = evaluate_with_fixed_threshold(
        loader=test_loader,
        backbone=backbone,
        classifier=classifier,
        device=device,
        threshold=best_threshold,
    )

    print_metrics(
        title="FINAL TEST RESULT",
        metrics=final_metrics,
    )


if __name__ == "__main__":
    main()