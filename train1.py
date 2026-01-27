import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# 确保你的模型代码在这里 (SSA_Model.py)
from SSA_Model import SSA_Model

# ================= 配置区 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
CSV_PATH = os.path.join(BASE_DIR, "metadata_multi.csv")
NPY_DIR = os.path.join(BASE_DIR, "coswara_multi_modal_npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-5  # SSA 架构较深，建议调小学习率
WEIGHT_DECAY = 0.05

BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_ssa_coswara_by_f1.pth")
CM_BEST_PATH = os.path.join(BASE_DIR, "confusion_matrix_best_by_f1.png")


# ================= 数据增强与工具 =================
def apply_spec_augment(spec, max_f=15, max_t=80):
    """针对频谱增强（假设 spec shape=[128,1024]）"""
    f = random.randint(0, max_f)
    f0 = random.randint(0, 128 - f) if (128 - f) > 0 else 0
    spec[f0:f0 + f, :] = 0

    t = random.randint(0, max_t)
    t0 = random.randint(0, 1024 - t) if (1024 - t) > 0 else 0
    spec[:, t0:t0 + t] = 0
    return spec


# ================= Dataset =================
class CoswaraDataset(Dataset):
    def __init__(self, df, npy_dir, is_train=False):
        self.npy_dir = npy_dir
        self.is_train = is_train
        self.samples = []

        # 核心逻辑：展开多模态文件
        for _, row in df.iterrows():
            user_id = row["user_id"]
            label = int(row["label"])
            modes = str(row["modes"]).split(",")

            for mode in modes:
                self.samples.append({
                    "npy_path": os.path.join(self.npy_dir, f"{user_id}_{mode}.npy"),
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec = np.load(sample["npy_path"])

        if self.is_train:
            spec = apply_spec_augment(spec)

        spec_t = torch.from_numpy(spec).float().unsqueeze(0)  # [1, 128, 1024]
        label = torch.tensor(sample["label"], dtype=torch.float32)  # scalar float
        return spec_t, label


# ================= 自动阈值：选 F1 最大的阈值 =================
def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray):
    """
    在 [0,1] 上搜索使 F1 最大的阈值（验证集）
    返回：best_thr, best_f1
    """
    # 为了速度与稳定性：用“概率的分位点阈值”搜索（比遍历 10000 个固定步长更合理）
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 201)))
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        cur_f1 = f1_score(y_true, preds)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


# ================= 主程序 =================
def main():
    df = pd.read_csv(CSV_PATH)

    # 1) 严格按 user_id 划分（防止数据泄露）
    unique_users = df["user_id"].unique()
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    train_ds = CoswaraDataset(train_df, NPY_DIR, is_train=True)
    val_ds = CoswaraDataset(val_df, NPY_DIR, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2) 初始化你的 SSA_Model
    model = SSA_Model(num_classes=1, n_layers=6, d_model=192, patch_time=1).to(DEVICE)

    # 3) 自动平衡损失权重（pos_weight = neg/pos）
    pos_count = float(train_df["label"].sum())
    neg_count = float(len(train_df) - train_df["label"].sum())
    if pos_count < 1:
        raise ValueError("训练集正样本为 0，无法计算 pos_weight。请检查数据/标签。")

    pos_weight = torch.tensor([neg_count / pos_count], device=DEVICE, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1_overall = -1.0  # ✅ 最优按 F1
    print(f"🚀 开始训练: 训练集包含 {len(train_ds)} 个模态样本, 验证集包含 {len(val_ds)} 个模态样本")
    print(f"📌 pos_weight = {pos_weight.item():.4f} (neg/pos) | device={DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        # ================= 训练 =================
        model.train()
        train_loss = 0.0

        for specs, labels in train_loader:
            specs = specs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).view(-1)  # [B]

            optimizer.zero_grad()
            logits = model(specs).view(-1)  # [B]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_loss = train_loss / max(len(train_loader), 1)

        # ================= 验证 =================
        model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True).view(-1)  # [B]

                logits = model(specs).view(-1)  # [B]
                probs = torch.sigmoid(logits)

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_labels = np.concatenate(all_labels).astype(int)  # [N]
        all_probs = np.concatenate(all_probs)                # [N]

        # AUC（与阈值无关）
        auc = roc_auc_score(all_labels, all_probs)

        # ✅ 自动阈值：选 F1 最大的阈值
        thr, f1_at_thr = best_threshold_by_f1(all_labels, all_probs)
        all_preds = (all_probs >= thr).astype(int)

        # 指标
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_at_thr  # 与上面一致
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"AUC: {auc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f} | "
            f"SE: {se:.4f} | SP: {sp:.4f} | Thr*: {thr:.3f}"
        )

        # ✅ best 按 F1 保存
        if f1 > best_f1_overall:
            best_f1_overall = f1
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print("⭐ New Best Model Saved (by F1)")

            # 保存混淆矩阵图
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"]
            )
            plt.title(f"Best Confusion Matrix\nF1={best_f1_overall:.4f}, Thr={thr:.3f}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(CM_BEST_PATH, dpi=200)
            plt.close()

    print(f"🎉 训练完成! 最高验证集 F1: {best_f1_overall:.4f}")
    print(f"✅ Best checkpoint saved to: {BEST_CKPT_PATH}")
    print(f"✅ Best confusion matrix saved to: {CM_BEST_PATH}")


if __name__ == "__main__":
    main()
