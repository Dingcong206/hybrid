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

# 这里假设你的 SSA_Model 定义在 SSA_Model.py 文件中
from SSA_Model import SSA_Model


# =====================================================
# 1) 损失函数：Focal Loss (针对医疗数据类别不平衡)
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1 - pt) ** self.gamma * BCE_loss).mean()


# =====================================================
# 2) 数据集类：读取 HeAR 提取好的 .npy
# =====================================================
class ICBHIDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 加载特征
        feat = np.load(row['feature_path']).astype(np.float32)
        # 统一形状：确保是 (97, 1024)，去掉可能存在的 CLS token
        if feat.shape[0] == 98:
            feat = feat[1:, :]

        label = torch.tensor(row['label'], dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 3) 指标计算：Se, Sp, ACC, AUC, ICBHI Score
# =====================================================
def compute_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # 鲁棒性处理：防止 cm 只有 1 个类别
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # 如果验证集太小或模型预测全是一类，手动补齐
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    icbhi_score = (se + sp) / 2
    return se, sp, acc, auc, icbhi_score, cm


# =====================================================
# 4) 训练主流程
# =====================================================
def train():
    # --- 参数配置 ---
    CSV_PATH = "/data/dingcong/hybrid/metadata_segmented.csv"
    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据加载与采样策略 ---
    dataset = ICBHIDataset(CSV_PATH)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    # 解决不平衡：Weighted Sampler
    train_labels = dataset.df.iloc[train_idx]['label'].values
    class_counts = np.bincount(train_labels.astype(int))
    weights = 1. / class_counts
    samples_weights = torch.from_numpy(weights[train_labels.astype(int)])
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    # --- 初始化 ---
    model = SSA_Model(input_dim=1024, d_model=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = FocalLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 训练循环 ---
    best_icbhi = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- 验证 ---
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                probs = torch.sigmoid(model(feats.to(DEVICE)))
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        se, sp, acc, auc, score, cm = compute_metrics(np.array(all_labels), np.array(all_probs))

        print(f"\n📊 [Epoch {epoch + 1}] Val Results:")
        print(f"Loss: {total_loss / len(train_loader):.4f} | Se: {se:.4f} | Sp: {sp:.4f}")
        print(f"Acc: {acc:.4f} | AUC: {auc:.4f} | ✨ ICBHI Score: {score:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        if score > best_icbhi:
            best_icbhi = score
            torch.save(model.state_dict(), "best_ssa_model.pth")
            print("🏆 New Best Score! Model Saved.")

        scheduler.step()


if __name__ == "__main__":
    train()