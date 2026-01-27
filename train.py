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

# 确保 VimA_Model.py 在同目录
from VimA_Model import VimAHybrid

# ================= 配置区 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # 如果 patch_time=1 导致 OOM，请调为 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

# 路径解析
PATIENT_PARSE_REGEX = r"^(\d+)"
BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_vima_patient_split.pth")
CM_BEST_PATH = os.path.join(BASE_DIR, "confusion_matrix_best.png")


# ================= 数据增强与工具 =================
def apply_spec_augment(spec, max_f=15, max_t=50):
    """ 简单的频谱增强：随机涂黑一段频率或时间 """
    # Frequency masking
    f = random.randint(0, max_f)
    f0 = random.randint(0, 128 - f)
    spec[f0:f0 + f, :] = 0
    # Time masking
    t = random.randint(0, max_t)
    t0 = random.randint(0, 1024 - t)
    spec[:, t0:t0 + t] = 0
    return spec


def compute_metrics(y_true, y_pred, y_prob):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {"acc": accuracy_score(y_true, y_pred), "se": se, "sp": sp, "icbhi": (se + sp) / 2,
            "auc": roc_auc_score(y_true, y_prob), "cm": cm}


# ================= Dataset =================
class ICBHIDataset(Dataset):
    def __init__(self, df, npy_dir, is_train=False):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.npy_dir, row["wav_name"].replace(".wav", ".npy"))
        spec = np.load(npy_path)  # (128, 1024)

        if self.is_train:
            spec = apply_spec_augment(spec)

        spec_t = torch.from_numpy(spec).float().unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.float)
        return spec_t, label


# ================= 主程序 =================
def main():
    df = pd.read_csv(CSV_PATH)
    # 解析 Patient ID
    df["patient_id"] = df["original_file"].apply(
        lambda x: re.match(PATIENT_PARSE_REGEX, str(x)).group(1) if re.match(PATIENT_PARSE_REGEX, str(x)) else str(x))

    # Patient Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR, is_train=True), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR, is_train=False), batch_size=BATCH_SIZE, num_workers=4,
                            pin_memory=True)

    # 初始化模型 (注意 patch_time=1 会显著增加序列长度)
    model = VimAHybrid(num_classes=1, d_model=192, patch_time=1).to(DEVICE)

    # 自动计算正样本权重
    pos = (train_df["label"] == 1).sum()
    neg = (train_df["label"] == 0).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_icbhi = 0
    print(f"开始训练: {len(train_df)} 训练样本, {len(val_df)} 验证样本 (按患者划分)")

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

        # 验证与寻找最佳阈值
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                probs = torch.sigmoid(model(specs))
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 在验证集寻找最佳 ICBHI 阈值
        best_thr_score = 0
        final_m = None
        for thr in np.arange(0.3, 0.7, 0.05):
            m = compute_metrics(all_labels, [1 if p > thr else 0 for p in all_probs], all_probs)
            if m["icbhi"] > best_thr_score:
                best_thr_score = m["icbhi"]
                final_m = m

        print(
            f"Epoch {epoch} | Loss: {train_loss / len(train_loader):.4f} | ICBHI: {final_m['icbhi']:.4f} (SE:{final_m['se']:.4f} SP:{final_m['sp']:.4f})")

        if final_m["icbhi"] > best_icbhi:
            best_icbhi = final_m["icbhi"]
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            # 保存混淆矩阵图
            plt.figure(figsize=(6, 5))
            sns.heatmap(final_m["cm"], annot=True, fmt="d", cmap="Blues")
            plt.title(f"Best ICBHI: {best_icbhi:.4f}")
            plt.savefig(CM_BEST_PATH)
            plt.close()
            print(f"⭐ 新纪录已保存")

    print(f"训练完成! 最高分: {best_icbhi:.4f}")


if __name__ == "__main__":
    main()