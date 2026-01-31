import sys
import os
from pathlib import Path

# 环境配置：确保能找到 mymodels 和 utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 导入自定义模块
from mymodels.model import build_model
from utils.metrics import user_metrics


# =====================================================
# 1) 增强型 Dataset：支持多段特征加载
# =====================================================
class CoswaraMultiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # 加载 (96, 1024) 的 HeAR 特征
            feat = np.load(row["feature_path"]).astype(np.float32)
            if feat.shape != (96, 1024):
                feat = np.resize(feat, (96, 1024)).astype(np.float32)
        except Exception:
            feat = np.zeros((96, 1024), dtype=np.float32)

        # 数据增强 (仅训练集)
        if self.is_train:
            # Time Masking
            if np.random.rand() < 0.4:
                t_mask = np.random.randint(5, 15)
                t0 = np.random.randint(0, 96 - t_mask)
                feat[t0: t0 + t_mask, :] = 0
            # Feature Masking
            if np.random.rand() < 0.3:
                f_mask = np.random.randint(50, 150)
                f0 = np.random.randint(0, 1024 - f_mask)
                feat[:, f0: f0 + f_mask] = 0

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label, row["user_id"]


# =====================================================
# 2) 损失函数：Focal Loss (处理类别不平衡)
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, label_smoothing=0.08):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, inputs, targets):
        # 标签平滑
        targets = targets * (1 - self.ls) + 0.5 * self.ls
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# =====================================================
# 3) 核心训练流程
# =====================================================
def train():
    # 路径配置
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_multi_segments.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH = str(OUT_DIR / "best_multi_user_f1.pth")

    # 超参数
    BATCH_SIZE = 64
    EPOCHS = 80
    MAX_LR = 1e-5
    WEIGHT_DECAY = 1e-2
    PATIENCE = 15
    MIN_SP = 0.65
    # 1. 加载数据并执行【用户级】划分 (严防泄露)
    df = pd.read_csv(CSV_PATH)
    unique_users = df[["user_id", "label"]].drop_duplicates()

    train_u, val_u = train_test_split(
        unique_users["user_id"],
        test_size=0.2,
        random_state=42,
        stratify=unique_users["label"]
    )

    train_df = df[df["user_id"].isin(train_u)]
    val_df = df[df["user_id"].isin(val_u)]

    # 2. DataLoader
    train_loader = DataLoader(
        CoswaraMultiDataset(train_df, is_train=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    # 验证集绝对不能 shuffle，否则 probs 顺序会乱
    val_loader = DataLoader(
        CoswaraMultiDataset(val_df, is_train=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 3. 初始化
    model = build_model(input_dim=1024, d_model=512, dropout=0.35).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_user_f1 = -1.0
    no_improve = 0

    print(f"🚀 开始训练 | 训练集用户: {len(train_u)} | 验证集用户: {len(val_u)}")

    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for feats, labels, _ in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            logits = model(feats).view(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step(epoch)

        # --- 验证阶段 (对接你的 user_metrics) ---
        model.eval()
        seg_probs = []
        with torch.no_grad():
            for feats, _, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                seg_probs.extend(torch.sigmoid(logits).cpu().numpy())

        seg_probs = np.asarray(seg_probs)

        # 调用你的 metrics.py 逻辑
        u_auc, u_best = user_metrics(
            val_df,
            seg_probs,
            mode="f1_sp",
            min_sp=MIN_SP
        )

        # --- 打印结果与混淆矩阵 ---
        print(f"\n" + "—" * 45)
        print(f"📈 [Epoch {epoch}] User-Level Result:")
        print(f"AUC: {u_auc:.4f} | F1: {u_best['f1']:.4f} | SE: {u_best['se']:.4f} | SP: {u_best['sp']:.4f}")
        print(f"Best Threshold: {u_best['thr']:.2f}")

        if u_best['cm'] is not None:
            tn, fp, fn, tp = u_best['cm'].ravel()
            print(f"\nConfusion Matrix:")
            print(f"             Pred Neg    Pred Pos")
            print(f"Actual Neg     {tn:<10}  {fp:<10}")
            print(f"Actual Pos     {fn:<10}  {tp:<10}")
        print("—" * 45 + "\n")

        # --- 保存逻辑 ---
        cur_f1 = float(u_best["f1"])

        if cur_f1 > best_user_f1:
            best_user_f1 = cur_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New Best User F1: {best_user_f1:.4f} (Saved to {SAVE_PATH})")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early Stopping! Best F1: {best_user_f1:.4f}")
            break


if __name__ == "__main__":
    train()