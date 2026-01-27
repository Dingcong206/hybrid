import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# 导入你刚才准备好的模型文件
from VimA_Model import VimAHybrid

# ================= 配置区 (Linux 路径) =================
BASE_DIR = "/data/dingcong/hybrid"
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4


# =====================================================

# 1. 定义数据加载工具
class ICBHIDataset(torch.utils.data.Dataset):
    def __init__(self, df, npy_dir):
        self.df = df
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 确保文件名对应
        npy_path = os.path.join(self.npy_dir, row['wav_name'].replace('.wav', '.npy'))

        # 加载数据
        spec = np.load(npy_path)  # (128, 1024)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)  # 变为 (1, 128, 1024)
        label = torch.tensor(row['label'], dtype=torch.float)
        return spec_t, label


# 2. 准备数据
print("正在加载数据索引...")
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR), batch_size=BATCH_SIZE)

# 3. 初始化模型、损失函数和优化器
print(f"正在 {DEVICE} 上初始化 VimA 模型...")
model = VimAHybrid(num_classes=1, d_model=192, patch_time=4).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
criterion = nn.BCEWithLogitsLoss()

# 4. 训练循环
print("开始训练...")
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for specs, labels in train_loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)

        # 前向传播
        outputs = model(specs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证环节
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)

            logits = model(specs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算各项指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    # 混淆矩阵 (tn, fp, fn, tp)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    # 计算 SE (Sensitivity / Recall) 和 SP (Specificity)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度（真阳率）
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度（真阴率）
    icbhi_score = (se + sp) / 2

    print(f"\n--- Epoch [{epoch + 1}] 详细评估 ---")
    print(f"ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"SE (Sensitivity): {se:.4f} | SP (Specificity): {sp:.4f}")
    print(f"ICBHI Score: {icbhi_score:.4f}")

    # 如果是最后一个 Epoch，保存混淆矩阵图片
    if epoch == EPOCHS - 1:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.savefig('confusion_matrix.png')
        print("📊 混淆矩阵图已保存至 confusion_matrix.png")

    # 根据 ICBHI Score 保存模型（比只看 Acc 更科学）
    if icbhi_score > best_acc:
        best_acc = icbhi_score
        torch.save(model.state_dict(), "best_vima_model.pth")
        print("⭐ 发现更高 ICBHI 分数的模型，已保存！")