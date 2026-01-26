import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, train_test_split
from VimA_Model import VimAHybrid
import pandas as pd
import os

# ================= 配置区 =================
CSV_PATH = r"D:\Python project\PythonProject\ICBHI\metadata.csv"
NPY_DIR = r"D:\Python project\PythonProject\ICBHI\Respiratory_Sound_Database\Respiratory_Sound_Database\spec_npy_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4


# ==========================================

# 1. 组装 Dataset (我们在前几步讨论过的逻辑)
class ICBHIDataset(torch.utils.data.Dataset):
    def __init__(self, df, npy_dir):
        self.df = df
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.npy_dir, row['wav_name'].replace('.wav', '.npy'))
        spec = np.load(npy_path)  # (128, 1024)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)  # (1, 128, 1024)
        label = torch.tensor(row['label'], dtype=torch.float)
        return spec_t, label


# 2. 准备数据加载器
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR), batch_size=BATCH_SIZE)

# 3. 初始化你的“创造物”
model = VimAHybrid(num_classes=1, d_model=192, patch_time=4).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
criterion = nn.BCEWithLogitsLoss()

# 4. 训练循环
print(f"开始在 {DEVICE} 上训练 VimA 模型...")
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for specs, labels in train_loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)

        outputs = model(specs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证环节
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            outputs = torch.sigmoid(model(specs))
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {train_loss / len(train_loader):.4f} Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_vima_model.pth")
        print("⭐ 发现更好的模型，已保存！")