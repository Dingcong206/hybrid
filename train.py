import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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
    correct, total = 0, 0
    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            outputs = torch.sigmoid(model(specs))
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {train_loss / len(train_loader):.4f} Val Acc: {val_acc:.4f}")

    # 保存最优模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_vima_model.pth")
        print("⭐ 发现更好的模型，已保存权重！")