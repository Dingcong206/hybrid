import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

# ==========================================
# 0. 路径修正逻辑
# ==========================================
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from mymodels.model import build_model
from mymodels.dataset import RespiratoryDataset

# ==========================================
# 1. 基础配置
# ==========================================
TOTAL_CSV_PATH = "/data/dingcong/hybrid/labels.csv"
FEAT_DIR = "/data/dingcong/hybrid/hear_16x256_fixed"
SAVE_DIR = "./checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 4  # 2 略小，如果显存够建议 4 或 8
LR = 5e-5  # 稍微调低学习率，防止 Mamba 梯度爆炸

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 数据切分与准备
# ==========================================
full_df = pd.read_csv(TOTAL_CSV_PATH)
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

# 计算正负样本比例，用于 Loss 加权
# 如果 labels 是 0 和 1，计算 (负样本数量 / 正样本数量)
pos_count = (train_df['label'] == 1).sum()
neg_count = (train_df['label'] == 0).sum()
# 重点：给负样本更大的关注
imbalance_ratio = torch.tensor([neg_count / pos_count]).to(device)
print(f"📊 训练集比例 - 正: {pos_count}, 负: {neg_count} | 建议 pos_weight: {imbalance_ratio.item():.2f}")

train_csv_tmp = "train_split_tmp.csv"
val_csv_tmp = "val_split_tmp.csv"
train_df.to_csv(train_csv_tmp, index=False)
val_df.to_csv(val_csv_tmp, index=False)

train_ds = RespiratoryDataset(csv_path=train_csv_tmp, feat_dir=FEAT_DIR)
val_ds = RespiratoryDataset(csv_path=val_csv_tmp, feat_dir=FEAT_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 3. 模型与改进的损失函数
# ==========================================
model = build_model(in_dim=256, d_model=256, n_layers=4).to(device)

# --- 改进点 1: 使用带权重的损失函数 ---
# 降低正样本权重，相当于变相增加对“误判正常人”的惩罚
criterion = nn.BCEWithLogitsLoss(pos_weight=imbalance_ratio * 0.8)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)  # 增加 weight_decay 防止过拟合
# 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler('cuda')


# ==========================================
# 4. 训练与验证函数
# ==========================================

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for feats, labels in pbar:
        feats, labels = feats.to(device), labels.to(device).float()

        with torch.amp.autocast('cuda'):
            file_logit, _ = model(feats)
            loss = criterion(file_logit.view(-1), labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)


def validate():
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in tqdm(val_loader, desc="[Valid]"):
            feats = feats.to(device)
            labels = labels.to(device).float()

            file_logit, _ = model(feats)
            prob = torch.sigmoid(file_logit)

            all_probs.extend(prob.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # --- 改进点 2: 自动寻找最佳阈值 ---
    # 不要死守 0.5，寻找让 (SE+SP)/2 最高的阈值
    best_thr = 0.5
    max_icbhi = -1
    best_metrics = {}

    for thr in np.arange(0.3, 0.8, 0.05):
        preds = (all_probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()

        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        icbhi = (se + sp) / 2.0

        if icbhi > max_icbhi:
            max_icbhi = icbhi
            best_thr = thr
            best_metrics = {
                "AUC": roc_auc_score(all_labels, all_probs),
                "F1": f1_score(all_labels, preds, zero_division=0),
                "SE": se, "SP": sp, "ICBHI": icbhi,
                "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                "Best_Thr": thr
            }

    return best_metrics


# ==========================================
# 5. 主训练循环
# ==========================================
best_icbhi = -1
print("🚀 开始训练...")

try:
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(epoch)
        metrics = validate()
        scheduler.step()

        print(
            f"✅ Epoch {epoch} 总结 (Thr:{metrics['Best_Thr']:.2f}): Loss: {avg_loss:.4f} | "
            f"ICBHI: {metrics['ICBHI']:.4f} | SE: {metrics['SE']:.4f} | SP: {metrics['SP']:.4f} | "
            f"AUC: {metrics['AUC']:.4f} | TP:{metrics['TP']} TN:{metrics['TN']} FP:{metrics['FP']} FN:{metrics['FN']}"
        )

        if metrics["ICBHI"] > best_icbhi:
            best_icbhi = metrics["ICBHI"]
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"🌟 新最优 ICBHI={best_icbhi:.4f} (SP 提升至 {metrics['SP']:.4f})")

finally:
    if os.path.exists(train_csv_tmp): os.remove(train_csv_tmp)
    if os.path.exists(val_csv_tmp): os.remove(val_csv_tmp)

print(" 训练结束!")