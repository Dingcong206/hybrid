# tools/train1.py
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("sys.path[0]  =", sys.path[0])

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ✅ 用最硬的导入方式（不依赖 __init__.py）
from mymodels.model import build_model
from utils.metrics import segment_metrics, user_metrics


class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["feature_path"]).astype(np.float32)
            if feat.shape[0] == 97:   # drop CLS
                feat = feat[1:, :]
            if feat.shape != (96, 1024):
                feat = np.resize(feat, (96, 1024)).astype(np.float32)
        except Exception:
            feat = np.zeros((96, 1024), dtype=np.float32)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


def train():
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    EPOCHS = 30
    MAX_LR = 2e-4
    WEIGHT_DECAY = 3e-4
    DROPOUT = 0.15
    PATIENCE = 8
    CLIP_GRAD = 1.0

    # ✅ 统一输出目录
    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH_AUC = str(OUT_DIR / "best_user_auc_onecycle.pth")
    SAVE_PATH_F1  = str(OUT_DIR / "best_user_f1_onecycle.pth")

    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )
    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    train_labels = train_df["label"].values.astype(int)
    counts = np.bincount(train_labels)
    ratio = counts[0] / max(counts[1], 1)
    print("Train label distribution (segment-level):", counts, f"| ratio 0:1 = {ratio:.2f}:1")

    train_loader = DataLoader(CoswaraDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(CoswaraDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(input_dim=1024, d_model=256, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=50.0
    )

    best_user_auc = -1.0
    best_user_f1 = -1.0
    no_improve = 0

    print(f"🚀 Start training (AdamW + OneCycleLR)")
    print(f"   train={len(train_df)} segs, val={len(val_df)} segs | epochs={EPOCHS}, max_lr={MAX_LR}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()

            if CLIP_GRAD is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        model.eval()
        seg_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                seg_probs.extend(torch.sigmoid(logits).cpu().numpy())

        seg_probs = np.asarray(seg_probs)

        s_auc, s_best = segment_metrics(val_df["label"].values.astype(int), seg_probs)
        u_auc, u_best = user_metrics(val_df, seg_probs)

        print(f"\n📌 [Epoch {epoch}] Segment-Level (thr@F1-max):")
        print(f"   AUC: {s_auc:.4f} | F1: {s_best['f1']:.4f} | ACC: {s_best['acc']:.4f} | "
              f"SE: {s_best['se']:.4f} | SP: {s_best['sp']:.4f} | Thr: {s_best['thr']:.2f}")
        print(f"   CM:\n{s_best['cm']}")

        print(f"\n📈 [Epoch {epoch}] User-Level (mean agg, thr@F1-max):")
        print(f"   AUC: {u_auc:.4f} | F1: {u_best['f1']:.4f} | ACC: {u_best['acc']:.4f} | "
              f"SE: {u_best['se']:.4f} | SP: {u_best['sp']:.4f} | Thr: {u_best['thr']:.2f}")
        print(f"   CM:\n{u_best['cm']}")

        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH_AUC)
            print(f"💾 best USER-AUC updated -> {best_user_auc:.4f} | saved: {SAVE_PATH_AUC}")

        if u_best["f1"] > best_user_f1:
            best_user_f1 = u_best["f1"]
            torch.save(model.state_dict(), SAVE_PATH_F1)
            print(f"💾 best USER-F1 updated  -> {best_user_f1:.4f} | saved: {SAVE_PATH_F1}")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"🛑 Early stop: {PATIENCE} epochs no USER-F1 improvement. "
                  f"best USER-AUC={best_user_auc:.4f}, best USER-F1={best_user_f1:.4f}")
            break


if __name__ == "__main__":
    train()
