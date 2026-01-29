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

# 假设你的 SSA_Model 定义在 SSA_Model.py 中
from SSA_Model import SSA_Model


# =====================================================
# 1) 损失函数：Focal Loss 处理类别不平衡
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
# 2) Dataset：适配 Coswara 的 User-level 读取
# =====================================================
class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 加载 HeAR 提取的 .npy 特征 (形状通常为 97, 1024)
        feat = np.load(row["feature_path"]).astype(np.float32)

        # 还原论文：HeAR 输出 97 个 token (1 CLS + 96 Patches)
        # 如果 SSA_Model 不需要 CLS，则截取后 96 个
        if feat.shape[0] == 97:
            feat = feat[1:, :]  # 变为 (96, 1024)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 3) 指标计算
# =====================================================
def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob > thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auc = 0.5

    return se, sp, acc, auc, f1, cm


# =====================================================
# 4) 训练主程序 (User-level Split)
# =====================================================
def train():
    # --- 配置路径 ---
    # 这里的 CSV 应该是你上一步提取特征后生成的 coswara_metadata_segmented.csv
    CSV_PATH = "/data/dingcong/hybrid/coswara_metadata_segmented.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 超参数 ---
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    SAVE_PATH = "best_hear_ssa_coswara.pth"

    # 1. 读取数据并按用户划分
    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()

    # 为了 Stratify，获取每个用户的标签（取该用户所有录音中标签的最大值）
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    #
    train_users, val_users = train_test_split(
        users,
        test_size=0.2,
        random_state=42,
        stratify=user_labels
    )

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    print(f"✅ 数据划分完成:")
    print(f"   训练集: {len(train_users)} 用户, {len(train_df)} 样本")
    print(f"   验证集: {len(val_users)} 用户, {len(val_df)} 样本")

    # 2. 构建 DataLoader 与加权采样
    train_ds = CoswaraDataset(train_df)
    val_ds = CoswaraDataset(val_df)

    # 处理类别不平衡
    train_labels = train_df["label"].values.astype(int)
    counts = np.bincount(train_labels)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 初始化模型
    # HeAR 特征是 1024 维，序列长度截断 CLS 后是 96
    model = SSA_Model(input_dim=1024, d_model=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = -1.0

    # 4. 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(DEVICE)
                logits = model(feats).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        se, sp, acc, auc, f1, cm = compute_metrics(all_labels, all_probs)

        print(f"\n📊 [Epoch {epoch}] Val Results:")
        print(f"   AUC: {auc:.4f} | F1: {f1:.4f} | SE: {se:.4f} | SP: {sp:.4f}")
        print(f"   Confusion Matrix:\n{cm}")

        # 保存最优模型（以 AUC 为准，更适合严重不平衡数据）
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"🏆 发现更好模型，已保存至 {SAVE_PATH}")

        scheduler.step()


if __name__ == "__main__":
    train()