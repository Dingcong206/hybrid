import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score


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
        # Stem：条带卷积
        self.proj = nn.Conv2d(1, d_model, kernel_size=(freq_bins, patch_time), stride=(freq_bins, patch_time))
        self.norm = nn.LayerNorm(d_model)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if n_layers > 1 else 0.0
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
        x = self.proj(x)                    # [B, d_model, 1, L]
        x = x.flatten(2).transpose(1, 2)    # [B, L, d_model]
        x = self.norm(x)

        lstm_out, _ = self.lstm(x)          # [B, L, 2*d_model]
        out = torch.mean(lstm_out, dim=1)   # [B, 2*d_model]
        return self.head(out).squeeze(-1)   # [B]


# ================= 2. 数据处理 (Dataset) =================
class ICBHIDataset(Dataset):
    def __init__(self, df, npy_dir):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.npy_dir, row["wav_name"].replace(".wav", ".npy"))
        spec = np.load(npy_path)  # 期望 (128, 1024)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)  # [1, 128, 1024]
        label = torch.tensor(row["label"], dtype=torch.float)  # 0/1
        return spec_t, label


# ================= 3. 指标计算 =================
def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0

def compute_metrics(y_true, y_pred, y_prob):
    """
    y_true: list/np (0/1)
    y_pred: list/np (0/1)
    y_prob: list/np ([0,1])
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = safe_div(tp, tp + fn)
    sp = safe_div(tn, tn + fp)
    icbhi = (se + sp) / 2.0

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # 验证集如果只出现一个类别，AUC无法定义
        auc = float("nan")

    return acc, f1, auc, se, sp, icbhi, cm, (tn, fp, fn, tp)


# ================= 4. 主训练逻辑 =================
def train():
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_loader = DataLoader(
        ICBHIDataset(train_df, NPY_DIR),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        ICBHIDataset(val_df, NPY_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = BiLSTMBaseline().to(DEVICE)

    # 若你确定平衡，可保持 1.0；否则建议用：pos_weight = neg/pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {DEVICE}")
    print(f"开始训练 Bi-LSTM Baseline (样本数: {len(df)})...")

    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for specs, labels in train_loader:
            specs = specs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # ---- Val ----
        model.eval()
        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits = model(specs)
                probs = torch.sigmoid(logits)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_preds.extend((probs > 0.5).long().cpu().numpy().tolist())

        acc, f1, auc, se, sp, score, cm, (tn, fp, fn, tp) = compute_metrics(
            all_labels, all_preds, all_probs
        )

        print(f"\n--- Epoch [{epoch}/{EPOCHS}] ---")
        print(f"Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Validation: ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"            SE: {se:.4f} | SP: {sp:.4f} | ICBHI Score: {score:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"(TN={tn}, FP={fp}, FN={fn}, TP={tp})")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_bilstm_baseline.pth")
            print(f"⭐ 发现更高 ICBHI Score ({best_score:.4f}) 的模型，已保存权重！")

    print(f"\n🏆 Bi-LSTM 最高 ICBHI Score: {best_score:.4f}")


if __name__ == "__main__":
    train()
