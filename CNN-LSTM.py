import os
import re
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

# ================= 配置区 =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
CSV_PATH = os.path.join(BASE_DIR, "metadata_multi.csv")
NPY_DIR = os.path.join(BASE_DIR, "coswara_multi_modal_npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 40  # Baseline 通常收敛较快
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_baseline_cnn_lstm.pth")
CM_BEST_PATH = os.path.join(BASE_DIR, "cm_baseline_cnn_lstm.png")


# ================= 1. 数据增强与 Dataset =================
def apply_spec_augment(spec, max_f=15, max_t=80):
    f = random.randint(0, max_f)
    f0 = random.randint(0, 128 - f) if (128 - f) > 0 else 0
    spec[f0:f0 + f, :] = 0
    t = random.randint(0, max_t)
    t0 = random.randint(0, 1024 - t) if (1024 - t) > 0 else 0
    spec[:, t0:t0 + t] = 0
    return spec


class CoswaraDataset(Dataset):
    def __init__(self, df, npy_dir, is_train=False):
        self.npy_dir = npy_dir
        self.is_train = is_train
        self.samples = []
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
        if self.is_train: spec = apply_spec_augment(spec)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return spec_t, label


# ================= 2. 自动阈值搜索 =================
def best_threshold_by_f1(y_true, y_prob):
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 101)))
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        cur_f1 = f1_score(y_true, preds)
        if cur_f1 > best_f1:
            best_f1, best_thr = cur_f1, float(thr)
    return best_thr, float(best_f1)


# ================= 3. CNN-LSTM 架构 =================
class CNN_LSTM_Model(nn.Module):
    def __init__(self, d_model=128):
        super(CNN_LSTM_Model, self).__init__()
        # 卷积层提取局部频谱特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        # 将 [64, 32, 256] 展平并输入 LSTM
        self.lstm = nn.LSTM(input_size=64 * 32, hidden_size=d_model, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        # [B, 1, 128, 1024] -> CNN -> [B, 64, 32, 256]
        x = self.cnn(x)
        # 转换维度匹配 LSTM: [B, Time, Features] -> [B, 256, 64*32]
        x = x.permute(0, 3, 1, 2).flatten(2)
        x, _ = self.lstm(x)
        # 取最后一个时间步进行分类
        return self.fc(x[:, -1, :])


# ================= 4. 训练主逻辑 =================
def main():
    df = pd.read_csv(CSV_PATH)
    unique_users = df["user_id"].unique()
    # 使用相同的 random_state=42 确保对比公平
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    train_df = df[df["user_id"].isin(train_users)]
    val_df = df[df["user_id"].isin(val_users)]

    train_loader = DataLoader(CoswaraDataset(train_df, NPY_DIR, is_train=True), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(CoswaraDataset(val_df, NPY_DIR, is_train=False), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4)

    model = CNN_LSTM_Model().to(DEVICE)

    pos = train_df["label"].sum()
    neg = len(train_df) - pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_f1_val = -1.0
    print(f"📡 Baseline CNN-LSTM 启动 | 显卡: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE).view(-1)
            optimizer.zero_grad()
            logits = model(specs).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # 验证
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                logits = model(specs.to(DEVICE)).view(-1)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())

        all_labels, all_probs = np.concatenate(all_labels), np.concatenate(all_probs)
        auc = roc_auc_score(all_labels, all_probs)
        thr, f1 = best_threshold_by_f1(all_labels, all_probs)

        print(
            f"Epoch [{epoch}/{EPOCHS}] Loss: {t_loss / len(train_loader):.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | Thr: {thr:.3f}")

        if f1 > best_f1_val:
            best_f1_val = f1
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print("⭐ 发现更好的 Baseline 模型，已保存。")


if __name__ == "__main__":
    main()