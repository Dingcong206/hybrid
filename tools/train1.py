import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

# ====== 项目根目录，保证能 import mymodels ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import build_model  # noqa


# =========================
# 1) Dataset: 读取 (200,48) patch
# =========================
class HearPatchDataset(Dataset):
    def __init__(self, df, patch_dir):
        self.items = []
        self.patch_dir = patch_dir

        for _, row in df.iterrows():
            path = os.path.join(patch_dir, row["file_name"])
            label = int(row["label"])

            x = np.load(path)  # (N,200,48)
            if x.ndim == 2:
                x = x[None, ...]

            for i in range(x.shape[0]):
                self.items.append((x[i], label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        if idx == 0:
            print("🧪 Dataset sample shape:", x.shape)
        return (
            torch.tensor(x, dtype=torch.float32),  # (200,48)
            torch.tensor(y, dtype=torch.float32)
        )


# =========================
# 2) Focal Loss（可选）
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# =========================
# 3) 阈值扫描：ICBHI = (SE+SP)/2
# =========================
def scan_threshold_icbhi(y_true, y_prob, thr_grid=None):
    if thr_grid is None:
        thr_grid = np.linspace(0.05, 0.95, 181)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best = (-1.0, 0.5, 0.0, 0.0, None)  # icbhi, thr, se, sp, cm

    for thr in thr_grid:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        se = tp / (tp + fn + 1e-9)
        sp = tn / (tn + fp + 1e-9)
        icbhi = 0.5 * (se + sp)

        if icbhi > best[0]:
            best = (float(icbhi), float(thr), float(se), float(sp), cm)

    return best


# =========================
# 4) 训练主函数
# =========================
def train():
    CSV_PATH = "/data/dingcong/hybrid/labels.csv"
    PATCH_DIR = "/data/dingcong/hybrid/hear_patch_final"
    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH = str(OUT_DIR / "best_icbhi_score.pth")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    # 超参
    SEQ_LEN = 200
    INPUT_DIM = 48
    D_MODEL = 256
    N_LAYERS = 2
    DROPOUT = 0.25

    BATCH_SIZE = 48
    EPOCHS = 80
    LR = 6e-5
    WD = 1e-3
    PATIENCE = 20
    CLIP = 1.0

    # 读 CSV
    df = pd.read_csv(CSV_PATH)
    print("DF columns:", df.columns.tolist())
    print(df.head(2))

    # patient-wise split（如果没有 patient_id，就从 file_name 解析）
    if "patient_id" not in df.columns:
        df["patient_id"] = df["file_name"].astype(str).str.split("_", n=1).str[0]

    patients = df["patient_id"].unique()
    p_labels = df.groupby("patient_id")["label"].max().reindex(patients).values.astype(int)

    train_p, val_p = train_test_split(
        patients, test_size=0.2, random_state=42, stratify=p_labels
    )
    train_df = df[df["patient_id"].isin(train_p)].copy()
    val_df = df[df["patient_id"].isin(val_p)].copy()

    print(f"Train patients: {len(train_p)} | Val patients: {len(val_p)}")
    print(f"Train segs    : {len(train_df)} | Val segs    : {len(val_df)}")

    train_loader = DataLoader(
        HearPatchDataset(train_df, PATCH_DIR, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        HearPatchDataset(val_df, PATCH_DIR, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 模型
    model = build_model(
        input_dim=INPUT_DIM, seq_len=SEQ_LEN, d_model=D_MODEL,
        n_layers=N_LAYERS, dropout=DROPOUT, num_classes=1
    ).to(DEVICE)

    # sanity forward
    x0, y0 = next(iter(train_loader))
    with torch.no_grad():
        logits0 = model(x0.to(DEVICE))
    print(f"Sanity check: x {tuple(x0.shape)} logits {tuple(logits0.shape)}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    criterion = FocalLoss(alpha=0.35, gamma=2.0)

    best_icbhi = -1.0
    no_improve = 0

    print("开始训练（保存指标：ICBHI score = (SE+SP)/2，阈值验证集扫描）...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        total_loss = 0.0

        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)  # (B,200,48)
            y = y.to(DEVICE, non_blocking=True)  # (B,)
            if epoch == 0:
                print("🔥 Batch x shape:", x.shape)
                print("🔥 Batch y shape:", y.shape)
                break

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).view(-1)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step(epoch)

        # ===== val =====
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                logits = model(x).view(-1)
                prob = torch.sigmoid(logits).cpu().numpy()
                y_prob.extend(prob.tolist())
                y_true.extend(y.numpy().tolist())

        # AUC
        try:
            auc = roc_auc_score(np.asarray(y_true).astype(int), np.asarray(y_prob))
        except Exception:
            auc = float("nan")

        icbhi, thr, se, sp, cm = scan_threshold_icbhi(y_true, y_prob)

        print(f"\n[Epoch {epoch}] AUC={auc:.4f} | ICBHI={icbhi:.4f} | SE={se:.4f} | SP={sp:.4f} | thr={thr:.2f}")
        print("Confusion Matrix:\n", cm)

        if icbhi > best_icbhi:
            best_icbhi = icbhi
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ New Best ICBHI: {best_icbhi:.4f} (epoch {epoch}) -> saved: {SAVE_PATH}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early Stopping! Best ICBHI: {best_icbhi:.4f}")
                break


if __name__ == "__main__":
    train()
