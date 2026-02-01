import sys
from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ========== 环境配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 你自己的模块（保持不变）
from mymodels.model import build_model
# 你已有的 metrics 可以保留（这里不依赖它）
from utils.metrics import user_metrics  # noqa: F401

# patch -> tokens
from data.patch_utils import patch_10_200_48_to_tokens


# ----------------------------
# 0) 复现
# ----------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================
# 1) Dataset：读取 patch (T,200,48) -> tokens (96,48)
# =====================================================
class HearPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, patch_dir: str, seq_len: int = 96):
        self.df = df.reset_index(drop=True)
        self.patch_dir = patch_dir
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["file_name"]  # 你这里已经是 .npy
        label = float(row["label"])

        patch_path = os.path.join(self.patch_dir, fname)
        patch = np.load(patch_path)  # (T,200,48) 其中 T 可变
        patch = torch.tensor(patch, dtype=torch.float32)

        x = patch_10_200_48_to_tokens(patch, seq_len=self.seq_len)  # (96,48)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


# =====================================================
# 2) Focal Loss（你原本那套）
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, label_smoothing=0.08):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ls = float(label_smoothing)

    def forward(self, inputs, targets):
        targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 3) ICBHI score 验证：阈值搜索，返回最佳 (SE+SP)/2
# =====================================================
@torch.no_grad()
def eval_icbhi(model, loader, device, thr_grid=None):
    model.eval()
    all_probs = []
    all_y = []

    for x, y in loader:
        x = x.to(device)             # (B,96,48)
        logits = model(x).view(-1)   # (B,)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.numpy())

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(int)

    if thr_grid is None:
        thr_grid = np.linspace(0.05, 0.95, 91)  # 0.05~0.95, step=0.01

    best = {
        "icbhi": -1.0,
        "se": 0.0,
        "sp": 0.0,
        "f1": 0.0,
        "thr": 0.5,
        "cm": None,
    }

    eps = 1e-9
    for thr in thr_grid:
        y_pred = (probs >= thr).astype(int)

        # confusion matrix: [[TN, FP],[FN, TP]]
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        se = tp / (tp + fn + eps)
        sp = tn / (tn + fp + eps)
        icbhi = 0.5 * (se + sp)

        precision = tp / (tp + fp + eps)
        recall = se
        f1 = 2 * precision * recall / (precision + recall + eps)

        if icbhi > best["icbhi"]:
            best.update({
                "icbhi": float(icbhi),
                "se": float(se),
                "sp": float(sp),
                "f1": float(f1),
                "thr": float(thr),
                "cm": np.array([[tn, fp], [fn, tp]], dtype=int),
            })

    return best


# =====================================================
# 4) Train：按 patient_id split + 用 ICBHI score 保存
# =====================================================
def train():
    seed_all(42)

    CSV_PATH = "/data/dingcong/hybrid/labels.csv"
    PATCH_DIR = "/data/dingcong/hybrid/hear_patch_final"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    # 超参数（你现在这套能跑通就先用）
    SEQ_LEN = 96
    INPUT_DIM = 48
    D_MODEL = 256

    BATCH_SIZE = 48
    EPOCHS = 80
    MAX_LR = 6e-5
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.35
    PATIENCE = 20
    CLIP_GRAD = 1.0

    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH_ICBHI = str(OUT_DIR / "best_icbhi_score.pth")

    # ---- 读表 ----
    df = pd.read_csv(CSV_PATH)
    print("DF columns:", df.columns.tolist())
    print(df.head(2))

    assert "file_name" in df.columns, "labels.csv 必须包含 file_name（.npy 文件名）"
    assert "label" in df.columns, "labels.csv 必须包含 label（0/1）"

    # ---- ICBHI: patient-wise split（从文件名解析 patient_id）----
    # 例如：101_1b1_Al_sc_Meditron.npy -> patient_id=101
    df["patient_id"] = df["file_name"].astype(str).str.split("_").str[0]
    df["user_id"] = df["patient_id"]  # 你原脚本的逻辑可继续沿用

    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )

    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    val_df = df[df["user_id"].isin(val_users)].reset_index(drop=True)

    print(f"Train patients: {len(train_users)} | Val patients: {len(val_users)}")
    print(f"Train segs    : {len(train_df)} | Val segs    : {len(val_df)}")

    # ---- DataLoader ----
    train_loader = DataLoader(
        HearPatchDataset(train_df, PATCH_DIR, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        HearPatchDataset(val_df, PATCH_DIR, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # ---- 模型 ----
    model = build_model(input_dim=INPUT_DIM, d_model=D_MODEL, dropout=DROPOUT).to(DEVICE)

    # ---- 损失/优化器/调度 ----
    criterion = FocalLoss(alpha=0.35, gamma=2.0, label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    # ---- sanity check：确保 forward 维度没问题 ----
    x0, y0 = next(iter(train_loader))
    with torch.no_grad():
        out0 = model(x0.to(DEVICE)).view(-1)
    print("Sanity check:", "x", tuple(x0.shape), "logits", tuple(out0.shape))

    best_icbhi = -1.0
    best_info = None
    no_improve = 0

    print("开始训练（保存指标：ICBHI score = (SE+SP)/2）...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for feats, labels in pbar:
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step(epoch)

        # ---- 验证：阈值搜索，取最佳 ICBHI score ----
        best = eval_icbhi(model, val_loader, DEVICE)

        print(
            f"\n[Epoch {epoch}] "
            f"ICBHI={(best['icbhi']):.4f} | SE={best['se']:.4f} | SP={best['sp']:.4f} "
            f"| F1={best['f1']:.4f} | thr={best['thr']:.2f}"
        )
        print(f"Confusion Matrix:\n{best['cm']}")

        # ---- 保存最优：看 ICBHI score ----
        if best["icbhi"] > best_icbhi:
            best_icbhi = best["icbhi"]
            best_info = best
            no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "config": {
                        "seq_len": SEQ_LEN,
                        "input_dim": INPUT_DIM,
                        "d_model": D_MODEL,
                        "dropout": DROPOUT,
                    },
                },
                SAVE_PATH_ICBHI
            )
            print(f"⭐ New Best ICBHI: {best_icbhi:.4f} -> saved to {SAVE_PATH_ICBHI}")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early Stopping! Best ICBHI: {best_icbhi:.4f} | best={best_info}")
            break


if __name__ == "__main__":
    train()
