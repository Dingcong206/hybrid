import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区 (与 VimA 保持一致) =================
BASE_DIR = "/data/dingcong/hybrid"
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4


# ================= 1. Bi-LSTM 模型架构 =================
class BiLSTMBaseline(nn.Module):
    def __init__(self, num_classes=1, d_model=128, n_layers=3, freq_bins=128, patch_time=4):
        super().__init__()
        # 前端：声学条带卷积 (Stem) - 与 VimA 保持一致
        self.proj = nn.Conv2d(1, d_model, kernel_size=(freq_bins, patch_time), stride=(freq_bins, patch_time))
        self.norm = nn.LayerNorm(d_model)

        # 核心：双向 LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 128, 1024]
        x = self.proj(x)  # -> [B, d_model, 1, L]
        x = x.flatten(2).transpose(1, 2)  # -> [B, L, d_model]
        x = self.norm(x)

        lstm_out, _ = self.lstm(x)  # -> [B, L, d_model * 2]

        # 全局平均池化
        out = torch.mean(lstm_out, dim=1)
        return self.head(out).squeeze(-1)


# ================= 2. 数据处理 (Dataset) =================
class ICBHIDataset(Dataset):
    def __init__(self, df, npy_dir):
        self.df = df
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.npy_dir, row['wav_name'].replace('.wav', '.npy'))
        spec = np.load(npy_path)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)
        label = torch.tensor(row['label'], dtype=torch.float)
        return spec_t, label


# ================= 3. 主训练逻辑 =================
def train():
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR), batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = BiLSTMBaseline().to(DEVICE)
    # 既然数据平衡，权重设为 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"开始训练 Bi-LSTM Baseline (样本数: {len(df)})...")
    best_score = 0

    for epoch in range(EPOCHS):
        model.train()
        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(specs), labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                logits = model(specs)
                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend((probs > 0.5).float().cpu().numpy())

        # 计算 ICBHI 指标
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)
        score = (se + sp) / 2

        print(f"Epoch [{epoch + 1}/{EPOCHS}] SE: {se:.4f} SP: {sp:.4f} Score: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_bilstm_baseline.pth")

    print(f"🏆 Bi-LSTM 最高 Score: {best_score:.4f}")


if __name__ == "__main__":
    train()