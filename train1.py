import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


# 导入你刚才定义的模型
# from SSA_Model import SSA_Model

# =====================================================
# 1) Focal Loss: 强迫模型关注难学的异常样本
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# =====================================================
# 2) Dataset: 加载你提取的 (97, 1024) 特征
# =====================================================
class ICBHIDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['feature_path']
        label = self.data.iloc[idx]['label']
        # 加载 HeAR 特征
        features = np.load(path).astype(np.float32)
        # 如果维度是 (98, 1024)，截断为 (97, 1024)
        if features.shape[0] == 98:
            features = features[1:, :]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)


# =====================================================
# 3) 核心训练与指标计算函数
# =====================================================
def calculate_icbhi_score(y_true, y_pred_bin, y_prob):
    cm = confusion_matrix(y_true, y_pred_bin)
    # 处理可能的单类别情况
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0  # 异常处理

    se = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity / Recall
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    acc = accuracy_score(y_true, y_pred_bin)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    score = (se + sp) / 2
    return se, sp, acc, auc, score, cm


def train():
    # --- 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    LR = 1e-4
    BATCH_SIZE = 64
    CSV_PATH = "/data/dingcong/hybrid/metadata_segmented.csv"

    # --- 数据准备 ---
    full_dataset = ICBHIDataset(CSV_PATH)

    # 建议按病人 ID 划分，这里简化演示使用随机划分
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)

    # 类别权重处理（针对数据不平衡）
    labels = [full_dataset.data.iloc[i]['label'] for i in train_idx]
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[int(t)] for t in labels]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型初始化 ---
    model = SSA_Model(input_dim=1024, d_model=256).to(device)
    criterion = FocalLoss()  # 使用 Focal Loss 提升敏感度
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 循环 ---
    best_score = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for feat, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(feat)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for feat, label in val_loader:
                feat = feat.to(device)
                prob = torch.sigmoid(model(feat))
                all_labels.extend(label.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs > 0.5).astype(int)

        se, sp, acc, auc, score, cm = calculate_icbhi_score(all_labels, all_preds, all_probs)

        print(f"\n📊 [Epoch {epoch + 1}] Val Results:")
        print(f"Loss: {train_loss / len(train_loader):.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
        print(f"Se (Recall): {se:.4f} | Sp: {sp:.4f}")
        print(f"✨ ICBHI Score: {score:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_ssa_model.pth")
            print("🏆 New Best Score! Model Saved.")

        scheduler.step()


if __name__ == "__main__":
    train()