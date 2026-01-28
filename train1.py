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

# 确保 SSA_Model.py 已经改成了我刚才给你的那个“厚重版”
from SSA_Model import SSA_Model

# ================= 配置区 =================
BASE_DIR = "/data/dingcong/hybrid"
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")  # 之前提取特征生成的 CSV
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128  # Embedding 很轻量，可以适当加大 Batch
EPOCHS = 50 # 既然收敛快，30足够
LEARNING_RATE = 1e-4  # 使用混合架构，可以从稍大的 LR 开始，配合 Scheduler
WEIGHT_DECAY = 0.01

BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_ssa_model.pth")


# ================= Dataset =================
class HeARDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 加载 HeAR 导出的 .npy 文件 (Shape: [Time, 768])
        spec = np.load(row["feature_path"])

        # 转换为 Tensor: [Time, 768]
        spec_t = torch.from_numpy(spec).float()
        label = torch.tensor(row["label"], dtype=torch.float32)
        return spec_t, label


# ================= 自动阈值函数 (保持不变) =================
def best_threshold_by_f1(y_true, y_prob):
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 101)))
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        cur_f1 = f1_score(y_true, preds)
        if cur_f1 > best_f1:
            best_f1, best_thr = cur_f1, float(thr)
    return best_thr, best_f1


# ================= 主程序 =================
def main():
    df = pd.read_csv(CSV_PATH)

    # 1) 划分数据集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_ds = HeARDataset(train_df)
    val_ds = HeARDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2) 初始化厚重版 SSA_Model
    # input_dim=768 (HeAR维度), d_model=192, n_layers=4 (每个Block内含3个Mamba)
    model = SSA_Model(input_dim=768, num_classes=1, n_layers=4, d_model=192, dropout=0.3).to(DEVICE)

    # 3) 损失函数 (带权重平衡)
    pos_weight = torch.tensor([(len(train_df) - train_df["label"].sum()) / train_df["label"].sum()], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 使用 CosineAnnealing 配合 Warmup（如果需要手动加）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1_overall = -1.0
    patience = 10  # 早停
    counter = 0

    print(f"🚀 开始训练: 样本总量 {len(df)} | 类别权重 {pos_weight.item():.2f}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # 验证逻辑
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                probs = torch.sigmoid(model(features))
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())

        all_labels = np.concatenate(all_labels).astype(int)
        all_probs = np.concatenate(all_probs)

        auc = roc_auc_score(all_labels, all_probs)
        thr, f1 = best_threshold_by_f1(all_labels, all_probs)

        print(
            f"Epoch [{epoch}/{EPOCHS}] Loss: {train_loss / len(train_loader):.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Thr: {thr:.3f}")

        if f1 > best_f1_overall:
            best_f1_overall = f1
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print("⭐ Saved Best Model")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early Stopping triggered.")
                break

    print(f"✅ 训练结束。最高 F1: {best_f1_overall:.4f}")


if __name__ == "__main__":
    main()