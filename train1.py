import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from SSA_Model import SSA_Model


# =====================================================
# 1) Focal Loss
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 2) Dataset：segment-level 读取 npy
# =====================================================
class ICBHISegDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        required = {"feature_path", "label"}
        miss = required - set(self.df.columns)
        if miss:
            raise ValueError(f"DataFrame 缺少列: {miss}, 当前列: {self.df.columns.tolist()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["feature_path"]).astype(np.float32)

        # 如果有 CLS token: (98,1024) -> (97,1024)
        if feat.ndim == 2 and feat.shape[0] == 98:
            feat = feat[1:, :]

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 3) Metrics
# =====================================================
def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    y_pred = (y_prob > thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    icbhi = 0.5 * (se + sp)
    return se, sp, acc, auc, icbhi, cm


# =====================================================
# 4) EarlyStopping
# =====================================================
class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -np.inf
        self.bad = 0

    def step(self, value: float) -> bool:
        if value > self.best + self.min_delta:
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# =====================================================
# 5) Train (recording-level split)
# =====================================================
def train():
    CSV_PATH = "/data/dingcong/hybrid/metadata_segmented.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 100
    WEIGHT_DECAY = 1e-5

    SAVE_PATH = "best_ssa_model_by_icbhi_recording_split.pth"
    PATIENCE = 30
    MIN_DELTA = 1e-4

    # ===== 读数据 =====
    df = pd.read_csv(CSV_PATH)

    required_cols = {"original_wav", "feature_path", "label"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"CSV 缺少列: {miss}, 当前 columns={df.columns.tolist()}")

    # ===== 录音级 label（用于 stratify）：该录音只要有一个异常 segment 就算异常录音 =====
    recs = df["original_wav"].unique()
    rec_label = df.groupby("original_wav")["label"].max().reindex(recs).values.astype(int)
    stratify = rec_label if len(np.unique(rec_label)) > 1 else None

    train_recs, val_recs = train_test_split(
        recs,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    train_df = df[df["original_wav"].isin(train_recs)].copy()
    val_df = df[df["original_wav"].isin(val_recs)].copy()

    print(f"✅ Recording split 完成：Train recs={len(train_recs)} | Val recs={len(val_recs)}")
    print(f"✅ Segments：Train={len(train_df)} | Val={len(val_df)}")
    print("📊 Train label dist:\n", train_df["label"].value_counts())
    print("📊 Val label dist:\n", val_df["label"].value_counts())

    # ===== DataLoader & Weighted Sampler（仅训练集）=====
    train_ds = ICBHISegDataset(train_df)
    val_ds = ICBHISegDataset(val_df)

    train_labels = train_df["label"].values.astype(int)
    class_counts = np.bincount(train_labels, minlength=2)
    class_counts = np.maximum(class_counts, 1)  # 防止除0
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ===== Model / Optim / Loss =====
    model = SSA_Model(input_dim=1024, d_model=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    early = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    best_icbhi = -1.0
    best_epoch = -1

    # ===== Train Loop =====
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            feats = feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)

        # ----- Val -----
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for feats, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                feats = feats.to(DEVICE, non_blocking=True)
                logits = model(feats).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

        se, sp, acc, auc, icbhi, cm = compute_metrics(all_labels, all_probs, thr=0.5)

        print(f"\n📊 [Epoch {epoch}] Val (recording-split, segment-eval)")
        print(f"Loss: {avg_loss:.4f} | SE: {se:.4f} | SP: {sp:.4f} | ICBHI: {icbhi:.4f}")
        print(f"ACC: {acc:.4f} | AUC: {auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Confusion Matrix:\n{cm}")

        # ----- Save best -----
        if icbhi > best_icbhi + MIN_DELTA:
            best_icbhi = icbhi
            best_epoch = epoch
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🏆 New Best ICBHI! Saved -> {SAVE_PATH} (best={best_icbhi:.4f} @ epoch {best_epoch})")

        # ----- Early stop -----
        if early.step(icbhi):
            print(f"\nEarly stopping：ICBHI 连续 {PATIENCE} 轮未提升（min_delta={MIN_DELTA}）。")
            print(f"✅ Best ICBHI = {best_icbhi:.4f} @ epoch {best_epoch}")
            break

        scheduler.step()

    print(f"\n✅ Done. Best ICBHI={best_icbhi:.4f} @ epoch {best_epoch}")
    print(f"📌 Best checkpoint: {SAVE_PATH}")


if __name__ == "__main__":
    train()
