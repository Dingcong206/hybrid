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

BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_ssa_coswara.pth")
CM_BEST_PATH = os.path.join(BASE_DIR, "confusion_matrix_coswara.png")


# ================= 数据增强与工具 =================
def apply_spec_augment(spec, max_f=15, max_t=80):
    """ 针对 1024 长度微调的频谱增强 """
    f = random.randint(0, max_f)
    f0 = random.randint(0, 128 - f)
    spec[f0:f0 + f, :] = 0
    t = random.randint(0, max_t)
    t0 = random.randint(0, 1024 - t)
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
            user_id = row['user_id']
            label = row['label']
            modes = row['modes'].split(',')
            for mode in modes:
                self.samples.append({
                    'npy_path': os.path.join(self.npy_dir, f"{user_id}_{mode}.npy"),
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec = np.load(sample['npy_path'])

        if self.is_train:
            spec = apply_spec_augment(spec)

        spec_t = torch.from_numpy(spec).float().unsqueeze(0)
        label = torch.tensor(sample['label'], dtype=torch.float)
        return spec_t, label


# ================= 主程序 =================
def main():
    df = pd.read_csv(CSV_PATH)

    # 1. 严格按 user_id 划分（防止数据泄露）
    unique_users = df['user_id'].unique()
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    train_df = df[df['user_id'].isin(train_users)]
    val_df = df[df['user_id'].isin(val_users)]

    train_ds = CoswaraDataset(train_df, NPY_DIR, is_train=True)
    val_ds = CoswaraDataset(val_df, NPY_DIR, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 初始化你的 SSA_Model (非对称架构)
    model = SSA_Model(num_classes=1, n_layers=6, d_model=192, patch_time=1).to(DEVICE)

    # 3. 自动平衡损失权重
    pos_count = train_df['label'].sum()
    neg_count = len(train_df) - pos_count
    # 针对 2:1 的比例，给正样本更高的权重
    pos_weight = torch.tensor([neg_count / pos_count], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0  # Coswara 任务通常看重 AUC
    print(f"🚀 开始训练: 训练集包含 {len(train_ds)} 个模态样本, 验证集包含 {len(val_ds)} 个模态样本")

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
        all_labels, all_probs = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                logits = model(specs.to(DEVICE))
                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # ===== 指标计算 (针对 COVID 任务优化) =====
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs > 0.5).astype(int)

        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)

        print(
            f"Epoch [{epoch}/{EPOCHS}] Loss: {train_loss / len(train_loader):.4f} | AUC: {auc:.4f} | SE: {se:.4f} | SP: {sp:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print("⭐ New Best Model Saved")

    print(f"🎉 训练完成! 最高验证集 AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()