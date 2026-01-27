import os
import re
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

from SSSA import VimAHybrid

# =========================
# 基本配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 0.05

BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_vima_patient.pth")
CM_SAVE_PATH = os.path.join(BASE_DIR, "confusion_matrix_best.png")

PATIENT_REGEX = r"^(\d+)"

# =========================
# Dataset（关闭数据增强）
# =========================
class ICBHIDataset(Dataset):
    """
    flip_label: 如果 True，则将原始 label 进行翻转： y = 1 - y
    目的：保证训练时的“正类=1”是少数类，从而 pos_weight 能正确起作用
    """
    def __init__(self, df, npy_dir, train=False, flip_label=False):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.train = train
        self.flip_label = flip_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_name = row["wav_name"].replace(".wav", ".npy")
        spec = np.load(os.path.join(self.npy_dir, npy_name))

        # ❌ 不做数据增强
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        y = float(row["label"])
        if self.flip_label:
            y = 1.0 - y

        # 让 label 形状与 logits 对齐：[1]
        label = torch.tensor([y], dtype=torch.float32)
        return spec, label


# =========================
# 阈值搜索：最大化 ICBHI=(SE+SP)/2
# =========================
def find_best_threshold(labels, probs, num=401):
    labels = np.array(labels).astype(int).reshape(-1)
    probs = np.array(probs).reshape(-1)

    best_thr, best_icbhi = 0.5, -1.0
    best_cm = None

    for thr in np.linspace(0.0, 1.0, num):
        preds = (probs > thr).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        icbhi = (se + sp) / 2

        if icbhi > best_icbhi:
            best_icbhi = icbhi
            best_thr = float(thr)
            best_cm = cm

    return best_thr, best_icbhi, best_cm


# =========================
# 主训练逻辑
# =========================
def train_proc():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    df["patient_id"] = df["original_file"].apply(
        lambda x: re.match(PATIENT_REGEX, str(x)).group(1) if re.match(PATIENT_REGEX, str(x)) else "unknown"
    )

    # 1) Patient split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    print("Label distribution (original):")
    print("  Train:", train_df["label"].value_counts().to_dict())
    print("  Val  :", val_df["label"].value_counts().to_dict())

    # 2) 自动判断：是否需要翻转标签以保证“1 是少数类”
    pos = int((train_df["label"] == 1).sum())
    neg = int((train_df["label"] == 0).sum())

    flip_label = False
    # 如果 label=1 不是少数类（pos >= neg），就翻转，让训练时的正类变成少数类
    if pos >= neg:
        flip_label = True
        # 翻转后，新的 pos/neg
        pos, neg = neg, pos

    # pos_weight 应该是 neg/pos（>1 才有意义）
    pos_weight_value = neg / (pos + 1e-8)

    print(f"flip_label = {flip_label} | pos={pos}, neg={neg} | pos_weight={pos_weight_value:.4f}")

    # 3) DataLoader（关闭增强：train=False/True 都不影响，因为 Dataset 里已经不做增强）
    train_loader = DataLoader(
        ICBHIDataset(train_df, NPY_DIR, train=True, flip_label=flip_label),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        ICBHIDataset(val_df, NPY_DIR, train=False, flip_label=flip_label),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 4) 模型
    model = VimAHybrid(num_classes=1, d_model=192, patch_time=4, num_layers=6).to(DEVICE)

    # ✅ 修正 pos_weight：让“训练时的正类=1”得到更大权重
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=DEVICE, dtype=torch.float32)
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = -1.0
    best_thr = 0.5

    print(f"Starting training: Train Samples {len(train_df)}, Val Samples {len(val_df)}")

    for epoch in range(1, EPOCHS + 1):
        # -------- Train --------
        model.train()
        total_loss = 0.0

        for specs, labels in train_loader:
            specs = specs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)  # [B,1]

            optimizer.zero_grad()
            logits = model(specs)  # 期望 [B,1]（若不是，请在这里 reshape）
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # -------- Val --------
        model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits = model(specs)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)

                probs = torch.sigmoid(logits)  # [B,1]

                all_labels.extend(labels.detach().cpu().numpy().reshape(-1))
                all_probs.extend(probs.detach().cpu().numpy().reshape(-1))

        # labels/probs 当前是“训练标签体系”（可能翻转过）
        # 指标计算也在这个体系下做（ICBHI/阈值选择一致），这样 best_thr 才能用于推理阶段
        # 如果你希望最后输出回“原始标签体系”，可以在输出时再反向翻转解释即可。

        # 1) 阈值搜索（最大化 ICBHI）
        thr, icbhi, cm = find_best_threshold(all_labels, all_probs, num=401)
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)

        # 2) 其他指标（可选）
        preds = (np.array(all_probs) > thr).astype(int)
        acc = accuracy_score(np.array(all_labels).astype(int), preds)

        # AUC：若 val 只有单一类别会报错，做个保护
        try:
            auc = roc_auc_score(np.array(all_labels).astype(int), np.array(all_probs))
        except ValueError:
            auc = float("nan")

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {total_loss / max(len(train_loader),1):.4f} | "
            f"ICBHI: {icbhi:.4f} | SE: {se:.4f} | SP: {sp:.4f} | "
            f"thr*: {thr:.3f} | ACC: {acc:.4f} | AUC: {auc:.4f}"
        )

        # 3) 保存最优（按 ICBHI）
        if icbhi > best_score:
            best_score = icbhi
            best_thr = thr
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "flip_label": flip_label,
                    "best_thr": best_thr,
                    "best_icbhi": best_score,
                    "pos_weight": pos_weight_value,
                },
                BEST_MODEL_PATH
            )

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Best ICBHI: {best_score:.4f} | thr*: {best_thr:.3f}")
            plt.xlabel("Pred")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(CM_SAVE_PATH)
            plt.close()
            print("⭐ Best model saved!")


if __name__ == "__main__":
    train_proc()
