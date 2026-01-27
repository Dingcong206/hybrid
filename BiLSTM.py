import os
import re
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区 (与 VimA 脚本完全对齐) =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # patch_time=1 会增加内存压力，建议 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

PATIENT_PARSE_REGEX = r"^(\d+)"
BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_bilstm_patient_split.pth")


# ================= 1. 数据增强与工具 =================
def apply_spec_augment(spec, max_f=15, max_t=50):
    f = random.randint(0, max_f)
    f0 = random.randint(0, 128 - f)
    spec[f0:f0 + f, :] = 0
    t = random.randint(0, max_t)
    t0 = random.randint(0, 1024 - t)
    spec[:, t0:t0 + t] = 0
    return spec


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    se = safe_div(tp, tp + fn)
    sp = safe_div(tn, tn + fp)
    return {"acc": accuracy_score(y_true, y_pred), "se": se, "sp": sp, "icbhi": (se + sp) / 2,
            "auc": roc_auc_score(y_true, y_prob), "cm": cm}


# ================= 2. Bi-LSTM 模型架构 =================
class BiLSTMBaseline(nn.Module):
    def __init__(self, num_classes=1, d_model=128, n_layers=3, freq_bins=128, patch_time=1):
        super().__init__()
        # 必须使用 patch_time=1 才能和 VimA 对齐输入的序列长度 (L=1024)
        self.proj = nn.Conv2d(1, d_model, kernel_size=(freq_bins, patch_time), stride=(freq_bins, patch_time))
        self.norm = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if n_layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, L, d_model]
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        out = torch.mean(lstm_out, dim=1)  # 全局平均池化
        return self.head(out).squeeze(-1)


# ================= 3. 数据处理 (Dataset) =================
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
        spec = np.load(npy_path)
        if self.is_train:
            spec = apply_spec_augment(spec)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.float)
        return spec_t, label


# ================= 4. 主训练逻辑 =================
def train():
    df = pd.read_csv(CSV_PATH)
    # 提取 Patient ID
    df["patient_id"] = df["original_file"].apply(
        lambda x: re.match(PATIENT_PARSE_REGEX, str(x)).group(1) if re.match(PATIENT_PARSE_REGEX, str(x)) else str(x))

    # Patient Split (必须保持 random_state=42)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR, is_train=True), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR, is_train=False), batch_size=BATCH_SIZE, num_workers=4)

    model = BiLSTMBaseline(patch_time=1).to(DEVICE)

    # 自动权重计算
    pos = (train_df["label"] == 1).sum()
    neg = (train_df["label"] == 0).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"开始训练 Bi-LSTM Patient-Split 版...")
    best_icbhi = -1.0

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
        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(specs)
                probs = torch.sigmoid(logits)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend((probs > 0.5).cpu().numpy())

        # ===== 计算指标 =====
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        icbhi = (se + sp) / 2

        print(
            f"\nEpoch [{epoch}/{EPOCHS}] "
            f"Loss: {train_loss / len(train_loader):.4f}\n"
            f"ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n"
            f"SE: {se:.4f} | SP: {sp:.4f} | ICBHI: {icbhi:.4f}"
        )

        # ===== 保存最优模型 =====
        if icbhi > best_icbhi:
            best_icbhi = icbhi
            torch.save(model.state_dict(), BEST_CKPT_PATH)

            # 保存混淆矩阵
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"]
            )
            plt.title(f"Best ICBHI = {best_icbhi:.4f}")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.tight_layout()
            plt.close()

            print("⭐ 新最优模型已保存")
if __name__ == "__main__":
    train()

