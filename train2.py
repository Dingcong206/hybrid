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

BEST_MODEL = "best_vima_patient.pth"
CM_PATH = "confusion_matrix_best.png"

PATIENT_REGEX = r"^(\d+)"


# =========================
# 数据增强
# =========================
def spec_augment(spec, max_f=15, max_t=50):
    f = random.randint(0, max_f)
    f0 = random.randint(0, spec.shape[0] - f)
    spec[f0:f0 + f, :] = 0

    t = random.randint(0, max_t)
    t0 = random.randint(0, spec.shape[1] - t)
    spec[:, t0:t0 + t] = 0
    return spec


# =========================
# Dataset
# =========================
class ICBHIDataset(Dataset):
    def __init__(self, df, npy_dir, train=False):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec = np.load(os.path.join(self.npy_dir, row["wav_name"].replace(".wav", ".npy")))

        if self.train:
            spec = spec_augment(spec)

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return spec, label


# =========================
# 指标
# =========================
def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn + 1e-8)
    sp = tn / (tn + fp + 1e-8)
    icbhi = (se + sp) / 2

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "se": se,
        "sp": sp,
        "icbhi": icbhi,
        "cm": cm
    }


# =========================
# 主训练逻辑
# =========================
def main():
    df = pd.read_csv(CSV_PATH)

    # 提取 patient id
    df["patient_id"] = df["original_file"].apply(
        lambda x: re.match(PATIENT_REGEX, str(x)).group(1)
    )

    # Patient split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = DataLoader(
        ICBHIDataset(train_df, NPY_DIR, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        ICBHIDataset(val_df, NPY_DIR, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 模型
    model = VimAHybrid(
        num_classes=1,
        d_model=192,
        patch_time=4,
        num_layers=6
    ).to(DEVICE)

    pos = (train_df["label"] == 1).sum()
    neg = (train_df["label"] == 0).sum()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], device=DEVICE)
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = 0

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0

        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # ================= 验证 =================
        model.eval()
        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(specs)
                probs = torch.sigmoid(logits)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend((probs > 0.5).cpu().numpy())

        # ===== 计算指标 =====
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        icbhi = (se + sp) / 2

        print(
            f"\nEpoch [{epoch}/{EPOCHS}] "
            f"Loss: {train_loss / len(train_loader):.4f}\n"
            f"ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n"
            f"SE: {se:.4f} | SP: {sp:.4f} | ICBHI: {icbhi:.4f}"
        )

        # ===== 保存最优模型 =====
        if icbhi > best_icbhi:
            best_icbhi = icbhi
            torch.save(model.state_dict(), BEST_CKPT_PATH)

            # 保存混淆矩阵
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"]
            )
            plt.title(f"Best ICBHI = {best_icbhi:.4f}")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.tight_layout()
            plt.savefig(CM_BEST_PATH)
            plt.close()

            print("⭐ 新最优模型已保存")
