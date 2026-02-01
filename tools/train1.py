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
from utils.metrics import user_metrics

# 你已经建好的 patch 转换函数
from data.patch_utils import patch_10_200_48_to_tokens


# =====================================================
# 1) Dataset：读取 patch (10,200,48) -> tokens (96,48)
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
        fname = row["file_name"]  # labels.csv 必须有这一列
        y = float(row["label"])

        patch_path = os.path.join(self.patch_dir, fname.replace(".wav", ".npy"))
        # 如果你 labels.csv 里 file_name 本来就是 .npy，那就用下面这一行替换上一行：
        # patch_path = os.path.join(self.patch_dir, fname)

        patch = np.load(patch_path)  # (10,200,48)
        patch = torch.tensor(patch, dtype=torch.float32)

        x = patch_10_200_48_to_tokens(patch, seq_len=self.seq_len)  # (96,48)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


# =====================================================
# 2) Focal Loss（你原来的 그대로）
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 3) Train（只保留一个 train()，修掉 user_id KeyError）
# =====================================================
def train():
    # 路径
    CSV_PATH = "/data/dingcong/hybrid/labels.csv"
    PATCH_DIR = "/data/dingcong/hybrid/hear_patch_final"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    # 超参数（你原本那套）
    BATCH_SIZE = 48
    EPOCHS = 80
    MAX_LR = 6e-5
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.35
    PATIENCE = 20
    CLIP_GRAD = 1.0

    MIN_SP = 0.65
    METRIC_MODE = "f1_sp"

    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH_AUC = str(OUT_DIR / "best_user_auc_v3.pth")
    SAVE_PATH_F1 = str(OUT_DIR / "best_user_f1_v3.pth")

    # ====== 读表 ======
    df = pd.read_csv(CSV_PATH)
    print("DF columns:", df.columns.tolist())
    print(df.head(2))

    # 必要列检查
    assert "file_name" in df.columns, "labels.csv 必须包含列：file_name（对应每条样本文件名）"
    assert "label" in df.columns, "labels.csv 必须包含列：label（0/1）"

    # ====== 关键：修复 KeyError: 'user_id' ======
    # 1) 如果没有 user_id，就用 file_name 兜底（每个文件算一个“用户组”）
    if "user_id" not in df.columns:
        df["user_id"] = df["file_name"]
        print("⚠️ labels.csv 没有 user_id，已用 file_name 作为 user_id 兜底（先跑通用）")

    # 2) 以 user 为单位做 stratify split（你原来的逻辑）
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )

    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    val_df = df[df["user_id"].isin(val_users)].reset_index(drop=True)

    print(f"Train users: {len(train_users)} | Val users: {len(val_users)}")
    print(f"Train segs : {len(train_df)} | Val segs : {len(val_df)}")

    # ====== DataLoader：用我们的 HearPatchDataset（别再用 CoswaraDataset）======
    train_loader = DataLoader(
        HearPatchDataset(train_df, PATCH_DIR, seq_len=96),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        HearPatchDataset(val_df, PATCH_DIR, seq_len=96),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # ====== 模型：input_dim=48（非常重要）======
    # 如果你的 build_model 需要 seq_len 参数，你也一起传：seq_len=96
    model = build_model(input_dim=48, d_model=256, dropout=DROPOUT).to(DEVICE)

    # 损失/优化器/调度
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.35, gamma=2.0, label_smoothing=0.08)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    best_user_auc = -1.0
    best_user_f1 = -1.0
    no_improve = 0

    # ====== (可选但强烈建议) 开训前先 forward 测试一次，避免再浪费时间 ======
    x0, y0 = next(iter(train_loader))
    with torch.no_grad():
        out0 = model(x0.to(DEVICE)).view(-1)
    print("Sanity check:", "x", tuple(x0.shape), "logits", tuple(out0.shape))

    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for feats, labels in pbar:
            feats = feats.to(DEVICE)    # (B,96,48)
            labels = labels.to(DEVICE)  # (B,)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step(epoch)

        # ====== 验证：收集 segment-level 概率 ======
        model.eval()
        seg_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                seg_probs.extend(torch.sigmoid(logits).cpu().numpy())

        seg_probs = np.asarray(seg_probs)

        # 你的 user_metrics 需要：
        # - 对应 val_df（里面要有 user_id / label / file_name 等）
        # - seg_probs 的顺序必须和 val_loader 取样顺序一致
        u_auc, u_best = user_metrics(val_df, seg_probs, mode=METRIC_MODE, min_sp=MIN_SP)

        print(f"\n[Epoch {epoch}] Val AUC: {u_auc:.4f} | F1: {u_best['f1']:.4f} | SP: {u_best['sp']:.4f}")
        print(f"Confusion Matrix:\n{u_best['cm']}")

        # 保存
        if u_best["f1"] > best_user_f1:
            best_user_f1 = u_best["f1"]
            torch.save(model.state_dict(), SAVE_PATH_F1)
            print(f"🌟 New Best F1: {best_user_f1:.4f}")
            no_improve = 0
        else:
            no_improve += 1

        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH_AUC)
            print(f"🔥 New Best AUC: {best_user_auc:.4f}")

        if no_improve >= PATIENCE:
            print(f"Early Stopping! Best AUC: {best_user_auc:.4f}, Best F1: {best_user_f1:.4f}")
            break


if __name__ == "__main__":
    train()
