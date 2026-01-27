import os
import re
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

# 确保 SSSA.py 里有 VimAHybrid
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

BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_vima_patient.pth")
CM_SAVE_PATH = os.path.join(BASE_DIR, "confusion_matrix_best.png")

PATIENT_REGEX = r"^(\d+)"


# =========================
# Dataset（不做增强 + 可选 label 翻转）
# =========================
class ICBHIDataset(Dataset):
    """
    flip_label:
      - False: 使用原始 label
      - True : 训练/验证时使用 y = 1 - y
    用途：保证训练时的 “1” 是少数类，pos_weight 才真正起作用。
    """
    def __init__(self, df, npy_dir, flip_label=False):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.flip_label = flip_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_name = str(row["wav_name"]).replace(".wav", ".npy")
        spec = np.load(os.path.join(self.npy_dir, npy_name))

        # 不做增强
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        y = float(row["label"])
        if self.flip_label:
            y = 1.0 - y

        # 返回 shape=[1]，保证跟 logits 对齐
        label = torch.tensor([y], dtype=torch.float32)
        return spec, label


# =========================
# 阈值搜索：最大化 ICBHI=(SE+SP)/2
# =========================
def find_best_threshold(y_true, y_prob, num=401):
    y_true = np.array(y_true).astype(int).reshape(-1)
    y_prob = np.array(y_prob).reshape(-1)

    best_thr = 0.5
    best_icbhi = -1.0
    best_cm = None
    best_se = 0.0
    best_sp = 0.0

    for thr in np.linspace(0.0, 1.0, num):
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        icbhi = (se + sp) / 2

        if icbhi > best_icbhi:
            best_icbhi = float(icbhi)
            best_thr = float(thr)
            best_cm = cm
            best_se = float(se)
            best_sp = float(sp)

    return best_thr, best_icbhi, best_se, best_sp, best_cm


# =========================
# 训练主流程
# =========================
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(NPY_DIR):
        raise FileNotFoundError(f"NPY_DIR not found: {NPY_DIR}")

    df = pd.read_csv(CSV_PATH)

    # patient_id
    def _pid(x):
        m = re.match(PATIENT_REGEX, str(x))
        return m.group(1) if m else "unknown"

    df["patient_id"] = df["original_file"].apply(_pid)

    # =========================
    # patient split
    # =========================
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    print("Label distribution (original):")
    print("  Train:", train_df["label"].value_counts().to_dict())
    print("  Val  :", val_df["label"].value_counts().to_dict())
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

    # =========================
    # 自动决定是否翻转 label（确保训练时 1 是少数类）
    # =========================
    pos_orig = int((train_df["label"] == 1).sum())
    neg_orig = int((train_df["label"] == 0).sum())

    flip_label = False
    if pos_orig >= neg_orig:
        flip_label = True
        # 翻转后 pos/neg 对调
        pos = neg_orig
        neg = pos_orig
    else:
        pos = pos_orig
        neg = neg_orig

    pos_weight_value = neg / (pos + 1e-8)

    print(f"flip_label = {flip_label} | pos={pos}, neg={neg} | pos_weight={pos_weight_value:.4f}")

    # =========================
    # DataLoader
    # =========================
    train_loader = DataLoader(
        ICBHIDataset(train_df, NPY_DIR, flip_label=flip_label),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        ICBHIDataset(val_df, NPY_DIR, flip_label=flip_label),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4
    )

    # =========================
    # Model + Loss
    # =========================
    model = VimAHybrid(
        num_classes=1,
        d_model=192,
        patch_time=4,
        num_layers=6
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=DEVICE, dtype=torch.float32)
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # =========================
    # Train
    # =========================
    best_icbhi = -1.0
    best_thr = 0.5

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for specs, labels in train_loader:
            specs = specs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)  # [B,1]

            optimizer.zero_grad()

            logits = model(specs)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)  # [B] -> [B,1]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        scheduler.step()

        # =========================
        # Validate
        # =========================
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

        # 1) 固定阈值 0.5 的指标（便于对照）
        y_true = np.array(all_labels).astype(int)
        y_prob = np.array(all_probs)
        y_pred_05 = (y_prob > 0.5).astype(int)

        cm_05 = confusion_matrix(y_true, y_pred_05, labels=[0, 1])
        tn, fp, fn, tp = cm_05.ravel()
        se_05 = tp / (tp + fn + 1e-8)
        sp_05 = tn / (tn + fp + 1e-8)
        icbhi_05 = (se_05 + sp_05) / 2

        acc_05 = accuracy_score(y_true, y_pred_05)
        f1_05 = f1_score(y_true, y_pred_05)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        # 2) 搜索最优阈值 thr*（最大化 ICBHI）
        thr_star, icbhi_star, se_star, sp_star, cm_star = find_best_threshold(y_true, y_prob, num=401)

        print(
            f"\nEpoch [{epoch}/{EPOCHS}] Loss: {total_loss / max(len(train_loader),1):.4f}\n"
            f"AUC: {auc:.4f}\n"
            f"@thr=0.5  ACC: {acc_05:.4f} | F1: {f1_05:.4f} | SE: {se_05:.4f} | SP: {sp_05:.4f} | ICBHI: {icbhi_05:.4f}\n"
            f"@thr*= {thr_star:.3f}               SE: {se_star:.4f} | SP: {sp_star:.4f} | ICBHI*: {icbhi_star:.4f}"
        )

        # =========================
        # Save best by ICBHI* (val)
        # =========================
        if icbhi_star > best_icbhi:
            icbhi_star = best_icbhi
            best_thr = thr_star

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "flip_label": flip_label,
                    "best_thr": best_thr,
                    "best_icbhi": best_icbhi,
                    "pos_weight": float(pos_weight_value),
                    "config": {
                        "d_model": 192,
                        "patch_time": 4,
                        "num_layers": 6,
                        "num_classes": 1
                    }
                },
                BEST_CKPT_PATH
            )

            # save confusion matrix (use thr*)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_star, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["0", "1"], yticklabels=["0", "1"])
            plt.title(f"Best ICBHI*: {best_icbhi:.4f} | thr*: {best_thr:.3f}")
            plt.xlabel("Pred")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(CM_SAVE_PATH)
            plt.close()

            print(f"⭐ Best saved! ICBHI*={best_icbhi:.4f}, thr*={best_thr:.3f}")
            print(f"   -> ckpt: {BEST_CKPT_PATH}")
            print(f"   -> cm  : {CM_SAVE_PATH}")


if __name__ == "__main__":
    main()
