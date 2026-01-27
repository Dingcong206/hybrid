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
from SSSA import VimAHybrid

# =========================
# 基本配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 100  # 增加上限，靠早停控制
LR = 1e-4
WEIGHT_DECAY = 0.05
EARLY_STOP_PATIENCE = 30  # 10轮不更新就停止

BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_vima_patient.pth")
CM_SAVE_PATH = os.path.join(BASE_DIR, "confusion_matrix_best.png")
PATIENT_REGEX = r"^(\d+)"


# ... [ICBHIDataset 和 find_best_threshold 函数保持不变] ...

class ICBHIDataset(Dataset):
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
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        y = float(row["label"])
        if self.flip_label: y = 1.0 - y
        return spec, torch.tensor([y], dtype=torch.float32)


def find_best_threshold(y_true, y_prob, num=401):
    y_true = np.array(y_true).astype(int).reshape(-1)
    y_prob = np.array(y_prob).reshape(-1)
    best_thr, best_icbhi, best_cm, best_se, best_sp = 0.5, -1.0, None, 0.0, 0.0
    for thr in np.linspace(0.0, 1.0, num):
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        icbhi = (se + sp) / 2
        if icbhi > best_icbhi:
            best_icbhi, best_thr, best_cm, best_se, best_sp = icbhi, thr, cm, se, sp
    return best_thr, best_icbhi, best_se, best_sp, best_cm


def main():
    # 数据加载与划分 [保持你原有的逻辑]
    df = pd.read_csv(CSV_PATH)
    df["patient_id"] = df["original_file"].apply(
        lambda x: re.match(PATIENT_REGEX, str(x)).group(1) if re.match(PATIENT_REGEX, str(x)) else "unknown")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

    # 自动翻转标签逻辑 [保持不变]
    pos_orig, neg_orig = int((train_df["label"] == 1).sum()), int((train_df["label"] == 0).sum())
    flip_label = pos_orig >= neg_orig
    pos, neg = (neg_orig, pos_orig) if flip_label else (pos_orig, neg_orig)
    pos_weight_value = neg / (pos + 1e-8)

    train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR, flip_label), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR, flip_label), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4)

    model = VimAHybrid(num_classes=1, d_model=192, patch_time=4, num_layers=6).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 训练核心变量 ---
    best_icbhi = -1.0
    epochs_no_improve = 0  # 早停计数器

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(specs).view(-1, 1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # --- 验证 ---
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                probs = torch.sigmoid(model(specs).view(-1, 1))
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_true, y_prob = np.array(all_labels).flatten(), np.array(all_probs).flatten()
        thr_star, icbhi_star, se_star, sp_star, cm_star = find_best_threshold(y_true, y_prob)

        # 计算辅助指标 (基于最优阈值)
        y_pred_star = (y_prob > thr_star).astype(int)
        acc_star = accuracy_score(y_true, y_pred_star)
        f1_star = f1_score(y_true, y_pred_star)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5

        print(f"\nEpoch [{epoch}/{EPOCHS}] | Loss: {train_loss / len(train_loader):.4f} | AUC: {auc:.4f}")
        print(
            f"ICBHI*: {icbhi_star:.4f} | SE: {se_star:.4f} | SP: {sp_star:.4f} | ACC: {acc_star:.4f} | F1: {f1_star:.4f} | Thr: {thr_star:.3f}")

        # --- 保存与早停逻辑 ---
        if icbhi_star > best_icbhi:
            best_icbhi = icbhi_star  # 修正：正确赋值给 best
            epochs_no_improve = 0  # 重置计数器

            torch.save({
                "model_state_dict": model.state_dict(),
                "best_icbhi": best_icbhi,
                "best_thr": thr_star,
                "flip_label": flip_label
            }, BEST_CKPT_PATH)

            # 保存混淆矩阵
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm_star, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Best ICBHI: {best_icbhi:.4f} (Epoch {epoch})")
            plt.savefig(CM_SAVE_PATH)
            plt.close()
            print(f"⭐ New Best Model Saved! (ICBHI: {best_icbhi:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"早停触发！连续 {EARLY_STOP_PATIENCE} 轮指标未提升。")
            break


if __name__ == "__main__":
    main()