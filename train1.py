import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 引入你改进后的 SSA_Model
from SSA_Model import SSA_Model


# =====================================================
# 1) 损失函数：带标签平滑的 Focal Loss
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, smoothing=0.05):  # 调高 alpha 至 0.5 提升特异度
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # 标签平滑：防止模型过度自信导致的阈值偏移
        with torch.no_grad():
            targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        inputs = inputs.view(-1)
        targets_s = targets_s.view(-1)

        bce = F.binary_cross_entropy_with_logits(inputs, targets_s, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 2) 评估辅助函数：增加 User-Level 聚合逻辑
# =====================================================
def get_user_metrics(df, probs):
    """
    按照 user_id 对预测概率进行平均聚合，计算用户级指标
    """
    df = df.copy()
    df['prob'] = probs
    # 按用户聚合：取平均概率
    user_res = df.groupby('user_id').agg({'prob': 'mean', 'label': 'max'}).reset_index()

    y_true = user_res['label'].values
    y_prob = user_res['prob'].values

    # 寻找 User-level 的最佳 F1 阈值
    best_f1 = 0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_true, (y_prob > thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob > best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return auc, best_f1, best_thr, cm


# =====================================================
# 3) 训练主程序
# =====================================================
def train():
    # --- 配置与超参数 ---
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    LR = 8e-5  # 略微降低学习率
    EPOCHS = 100
    WEIGHT_DECAY = 5e-4  # 显著增加权重衰减，抑制过拟合
    PATIENCE = 6  # 早停耐心值

    # 1. 数据准备 (User-level Split)
    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    # DataLoader 处理... (此处省略 Dataset 定义，同之前)
    # ... 使用 WeightedRandomSampler 处理类别不平衡 ...

    # 2. 初始化模型与优化器
    model = SSA_Model(input_dim=1024, d_model=256, dropout=0.3).to(DEVICE)  # 增加 Dropout
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.5, smoothing=0.1)

    # 引入 Warmup 调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_user_auc = 0
    early_stop_count = 0

    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # --- 验证阶段 ---
        model.eval()
        all_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        # 计算 User-level 指标
        u_auc, u_f1, u_thr, u_cm = get_user_metrics(val_df, all_probs)

        print(f"\n📊 [Epoch {epoch}] User-Level AUC: {u_auc:.4f} | F1: {u_f1:.4f} | Thr: {u_thr:.2f}")
        print(f"   CM:\n{u_cm}")

        # --- 早停与保存逻辑 ---
        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), "best_user_model.pth")
            print(f"⭐ 发现最佳 User AUC: {u_auc:.4f}，模型已保存")
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= PATIENCE:
            print(f"🛑 连续 {PATIENCE} 轮未提升，触发早停。最终最佳 User AUC: {best_user_auc:.4f}")
            break

        scheduler.step()


if __name__ == "__main__":
    train()