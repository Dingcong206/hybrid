import sys
import os
from pathlib import Path

# 环境配置
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
# 1) 增强型 Dataset：包含双向 Mask
# =====================================================
class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["feature_path"]).astype(np.float32)
            # 兼容 HeAR 原始输出
            if feat.shape[0] == 97:
                feat = feat[1:, :]
            if feat.shape != (96, 1024):
                feat = np.resize(feat, (96, 1024)).astype(np.float32)
        except Exception:
            feat = np.zeros((96, 1024), dtype=np.float32)

        # 数据增强 (仅在训练集)
        if self.is_train:
            # A. 时间轴 Mask (Time Masking)
            if np.random.rand() < 0.4:
                t_mask = np.random.randint(5, 15)
                t0 = np.random.randint(0, 96 - t_mask)
                feat[t0: t0 + t_mask, :] = 0

            # B. 特征轴 Mask (Feature Masking)
            if np.random.rand() < 0.3:
                f_mask = np.random.randint(50, 150)
                f0 = np.random.randint(0, 1024 - f_mask)
                feat[:, f0: f0 + f_mask] = 0

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 2) 损失函数：Focal Loss (解决类别不平衡)
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=2.0, label_smoothing=0.05):
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
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_avaigolable() else "cpu")

    # 配置超参数
    BATCH_SIZE = 48  # 略微减小 batch 让梯度更有随机性
    EPOCHS = 80  # 增加轮次，配合重启策略
    MAX_LR = 6e-5  # 降低学习率，针对 SSA 架构微调
    WEIGHT_DECAY = 1e-3  # 显著加大 L2 正则化
    DROPOUT = 0.35
    PATIENCE = 20  # 容忍度
    CLIP_GRAD = 1.0

    MIN_SP = 0.65  # 特异度底线
    METRIC_MODE = "f1_sp"

    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH_AUC = str(OUT_DIR / "best_user_auc_v3.pth")
    SAVE_PATH_F1 = str(OUT_DIR / "best_user_f1_v3.pth")

    # 加载数据
    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )

    train_loader = DataLoader(
        CoswaraDataset(df[df["user_id"].isin(train_users)], is_train=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        CoswaraDataset(df[df["user_id"].isin(val_users)], is_train=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 模型初始化 (推荐 d_model=512)
    model = build_model(input_dim=1024, d_model=512, dropout=DROPOUT).to(DEVICE)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    # 使用 Focal Loss 替换 BCE
    criterion = FocalLoss(alpha=0.35, gamma=2.0, label_smoothing=0.08)

    # 学习率调度：余弦退火热重启，有助于跳出局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    best_user_auc = -1.0
    best_user_f1 = -1.0
    no_improve = 0

    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # 调整学习率
        scheduler.step(epoch)

        # 验证阶段
        model.eval()
        seg_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                seg_probs.extend(torch.sigmoid(logits).cpu().numpy())

        seg_probs = np.asarray(seg_probs)
        u_auc, u_best = user_metrics(df[df["user_id"].isin(val_users)], seg_probs, mode=METRIC_MODE, min_sp=MIN_SP)

        print(f"\n[Epoch {epoch}] Val AUC: {u_auc:.4f} | F1: {u_best['f1']:.4f} | SP: {u_best['sp']:.4f}")
        print(f"Confusion Matrix:\n{u_best['cm']}")

        # 监控保存逻辑
        improved = False
        if u_best["f1"] > best_user_f1:
            best_user_f1 = u_best["f1"]
            torch.save(model.state_dict(), SAVE_PATH_F1)
            print(f"🌟 New Best F1: {best_user_f1:.4f}")
            no_improve = 0
            improved = True
        else:
            no_improve += 1

        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH_AUC)
            print(f"🔥 New Best AUC: {best_user_auc:.4f}")
            improved = True

        if no_improve >= PATIENCE:
            print(f"Early Stopping! Best AUC: {best_user_auc:.4f}, Best F1: {best_user_f1:.4f}")
            break


if __name__ == "__main__":
    train()