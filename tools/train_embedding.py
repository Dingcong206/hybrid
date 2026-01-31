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
# 1) Dataset：增加 user_id 返回，用于验证聚合
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
            feat = np.load(row["feature_path"]).astype(np.float32)
            if feat.shape != (96, 1024):
                feat = np.resize(feat, (96, 1024)).astype(np.float32)
        except Exception:
            feat = np.zeros((96, 1024), dtype=np.float32)

        # 训练集数据增强：双向 Masking
        if self.is_train:
            if np.random.rand() < 0.4:
                t_mask = np.random.randint(5, 15)
                t0 = np.random.randint(0, 96 - t_mask)
                feat[t0: t0 + t_mask, :] = 0
            if np.random.rand() < 0.3:
                f_mask = np.random.randint(50, 150)
                f0 = np.random.randint(0, 1024 - f_mask)
                feat[:, f0: f0 + f_mask] = 0

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        # 返回 user_id 字符串，方便验证时分组
        return torch.from_numpy(feat), label, row["user_id"]


# =========================
# 2) 损失函数：Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, label_smoothing=0.08):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.ls) + 0.5 * self.ls
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# =========================
# 3) 训练主函数
# =========================
def train():
    # 注意这里使用你新生成的多段 CSV
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_multi_segments.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    BATCH_SIZE = 64  # 片段多，可以稍微增大 batch
    EPOCHS = 80
    MAX_LR = 5e-5
    WEIGHT_DECAY = 1e-3
    PATIENCE = 15

    # 加载数据并按【用户】划分
    df = pd.read_csv(CSV_PATH)
    unique_users = df[["user_id", "label"]].drop_duplicates()

    # 严格按照用户 ID 进行划分，防止数据泄露
    train_u, val_u = train_test_split(
        unique_users["user_id"],
        test_size=0.2,
        random_state=42,
        stratify=unique_users["label"]
    )

    train_df = df[df["user_id"].isin(train_u)]
    val_df = df[df["user_id"].isin(val_u)]

    train_loader = DataLoader(CoswaraMultiDataset(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(CoswaraMultiDataset(val_df, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 模型与优化器
    model = build_model(input_dim=1024, d_model=512, dropout=0.35).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_auc = -1.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for feats, labels, _ in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(feats).view(-1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # --- 验证阶段：多段预测聚合 ---
        model.eval()
        results = []  # 存储 (user_id, prob, label)

        with torch.no_grad():
            for feats, labels, uids in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                for i in range(len(uids)):
                    results.append({"user_id": uids[i], "prob": probs[i], "label": labels[i].item()})

        # 将片段预测聚合为用户预测
        res_df = pd.DataFrame(results)
        user_pred_df = res_df.groupby("user_id").agg({"prob": "mean", "label": "first"}).reset_index()

        # 计算指标 (使用你原本的 user_metrics 函数)
        u_auc, u_best = user_metrics(
            val_df,  # 这里传入原验证集df供函数内部关联
            res_df["prob"].values,  # 传入片段级概率，user_metrics内部通常会处理聚合
            mode="f1_sp",
            min_sp=0.65
        )

        print(f"Epoch {epoch} | User AUC: {u_auc:.4f} | F1: {u_best['f1']:.4f}")

        if u_auc > best_auc:
            best_auc = u_auc
            torch.save(model.state_dict(), "best_multi_model.pth")
            no_improve = 0
            print("🔥 New Best AUC!")
        else:
            no_improve += 1

        if no_improve >= PATIENCE: break


if __name__ == "__main__":
    train()