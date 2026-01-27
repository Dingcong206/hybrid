import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 复用你之前的 Dataset 和 增强函数
from train import CoswaraDataset, apply_spec_augment, best_threshold_by_f1


# ================= 架构：CNN-LSTM =================
class CNN_LSTM_Baseline(nn.Module):
    def __init__(self, num_classes=1, d_model=128):
        super(CNN_LSTM_Baseline, self).__init__()

        # 1. CNN 特征提取层 (模拟频谱处理)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # [32, 64, 512]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # [64, 32, 256]
        )

        # 2. 桥接层：将 CNN 输出展平为序列
        # 输入是 [Batch, 64, 32, 256]，我们需要将其转为 [Batch, Time_Steps, Hidden]
        # 我们把 256 看作时间维度
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 256))

        # 3. LSTM 层 (双向)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 4. 分类头
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 1, 128, 1024]
        x = self.conv_block(x)  # [Batch, 64, 32, 256]

        # 压缩频率轴，保留时间轴
        x = torch.mean(x, dim=2)  # [Batch, 64, 256]
        x = x.transpose(1, 2)  # [Batch, 256, 64] -> [Batch, Time, Features]

        # LSTM 运行
        lstm_out, _ = self.lstm(x)  # [Batch, 256, d_model * 2]

        # 取最后一个时间步或全局池化
        x = torch.mean(lstm_out, dim=1)  # [Batch, d_model * 2]

        logits = self.fc(x)
        return logits


# ================= 配置与路径 (保持一致) =================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
CSV_PATH = os.path.join(BASE_DIR, "metadata_multi.csv")
NPY_DIR = os.path.join(BASE_DIR, "coswara_multi_modal_npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. 加载数据
    df = pd.read_csv(CSV_PATH)
    from sklearn.model_selection import train_test_split
    unique_users = df["user_id"].unique()
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    train_df = df[df["user_id"].isin(train_users)]
    val_df = df[df["user_id"].isin(val_users)]

    train_loader = DataLoader(CoswaraDataset(train_df, NPY_DIR, is_train=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(CoswaraDataset(val_df, NPY_DIR, is_train=False), batch_size=32, shuffle=False)

    # 2. 初始化模型
    model = CNN_LSTM_Baseline(num_classes=1).to(DEVICE)

    # 3. 损失函数与优化器
    pos_count = float(train_df["label"].sum())
    neg_count = float(len(train_df) - train_df["label"].sum())
    pos_weight = torch.tensor([neg_count / pos_count], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # LSTM 适合用常规 Adam

    # 4. 训练循环 (简化版)
    best_f1 = 0
    for epoch in range(1, 31):  # Baseline 跑 30 轮通常就够了
        model.train()
        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE).view(-1)
            optimizer.zero_grad()
            loss = criterion(model(specs).view(-1), labels)
            loss.backward()
            optimizer.step()

        # 验证逻辑
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                probs = torch.sigmoid(model(specs.to(DEVICE)).view(-1))
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        auc = roc_auc_score(all_labels, all_probs)
        thr, f1 = best_threshold_by_f1(all_labels, all_probs)

        print(f"Epoch {epoch} | AUC: {auc:.4f} | F1: {f1:.4f} | Thr: {thr:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_cnn_lstm_baseline.pth")


if __name__ == "__main__":
    main()