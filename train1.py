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

# 确保你的 SSA_Model.py 就在同级目录下
from SSA_Model import SSA_Model


# =====================================================
# 1) Dataset：适配 HeAR 特征读取
# =====================================================
class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 加载特征 (97, 1024)
        try:
            feat = np.load(row["feature_path"]).astype(np.float32)
            # 还原论文：HeAR 输出 97 个 token，去掉第一个 CLS 留 96 个给 SSA
            if feat.shape[0] == 97:
                feat = feat[1:, :]
        except Exception as e:
            # 容错处理
            feat = np.zeros((96, 1024), dtype=np.float32)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 2) 损失函数：带标签平滑的 Focal Loss (解决阈值偏移)
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # 标签平滑
        with torch.no_grad():
            targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        inputs = inputs.view(-1)
        targets_s = targets_s.view(-1)

        bce = F.binary_cross_entropy_with_logits(inputs, targets_s, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 3) User-Level 指标计算 (这是你最关心的指标)
# =====================================================
def get_user_metrics(df, probs):
    temp_df = df.copy()
    temp_df['prob'] = probs
    # 按用户聚合平均分
    user_res = temp_df.groupby('user_id').agg({'prob': 'mean', 'label': 'max'}).reset_index()

    y_true = user_res['label'].values
    y_prob = user_res['prob'].values

    # 自动搜索最佳 F1 阈值
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_true, (y_prob > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob > best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # 计算 SE/SP
    tn, fp, fn, tp = cm.ravel()
    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0

    return auc, best_f1, best_thr, se, sp, cm


# =====================================================
# 4) 训练主程序
# =====================================================
def train():
    # --- 超参数配置 ---
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    LR = 8e-5
    EPOCHS = 50
    WEIGHT_DECAY = 5e-4  # 强正则化，抑制过拟合
    PATIENCE = 6  # 早停耐受度
    SAVE_PATH = "best_ssa_user_model.pth"

    # 1. 加载与划分数据
    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    # 2. 构建 DataLoader
    train_ds = CoswaraDataset(train_df)
    val_ds = CoswaraDataset(val_df)

    # 类别平衡采样
    train_labels = train_df["label"].values.astype(int)
    counts = np.bincount(train_labels)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("Train label distribution:", np.bincount(train_labels))

    # 3. 初始化
    model = SSA_Model(input_dim=1024, d_model=256, dropout=0.3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.5, smoothing=0.1)
    # 带重启的余弦退火，防止陷入局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_user_auc = 0
    early_stop_counter = 0

    print(f"🚀 开始训练! 训练样本: {len(train_df)}, 验证样本: {len(val_df)}")

    for epoch in range(1, EPOCHS + 1):
        # --- 训练轮次 ---
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 验证轮次 ---
        model.eval()
        all_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        # 计算 User-level 指标
        u_auc, u_f1, u_thr, u_se, u_sp, u_cm = get_user_metrics(val_df, all_probs)

        print(f"\n📈 [Epoch {epoch}] User-Level Result:")
        print(f"   AUC: {u_auc:.4f} | F1: {u_f1:.4f} | SE: {u_se:.4f} | SP: {u_sp:.4f} | Thr: {u_thr:.2f}")
        print(f"   Confusion Matrix:\n{u_cm}")

        # --- 保存与早停逻辑 ---
        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ 发现更好的 User-AUC，已保存模型！")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print(f"🛑 连续 {PATIENCE} 轮没有提升，触发早停。最终最佳 User-AUC: {best_user_auc:.4f}")
            break

        scheduler.step()


if __name__ == "__main__":
    train()