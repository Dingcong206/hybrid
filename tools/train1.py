import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score,confusion_matrix
from tqdm import tqdm


# ==========================================
# 0. 路径修正逻辑
# ==========================================
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

# 现在可以正确导入了
from mymodels.model import build_model
from mymodels.dataset import RespiratoryDataset

# ==========================================
# 1. 基础配置
# ==========================================
# --- 请根据你的实际情况修改以下路径 ---
TOTAL_CSV_PATH = "/data/dingcong/hybrid/labels.csv" # 你的总CSV路径
FEAT_DIR = "/data/dingcong/hybrid/hear_16x256_fixed"           # 你的特征文件夹
SAVE_DIR = "./checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 2
LR = 1e-4

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 数据切分与准备
# ==========================================
# 读取总表
full_df = pd.read_csv(TOTAL_CSV_PATH)

# 自动划分：80% 训练, 20% 验证 (random_state 保证实验可重复)
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

# 为了配合你之前写的 Dataset 类（它接收 CSV 路径），我们生成两个临时 CSV
train_csv_tmp = "train_split_tmp.csv"
val_csv_tmp = "val_split_tmp.csv"
train_df.to_csv(train_csv_tmp, index=False)
val_df.to_csv(val_csv_tmp, index=False)

print(f"📊 数据划分完成: 训练集 {len(train_df)} 样本, 验证集 {len(val_df)} 样本")

# 实例化 Dataset
train_ds = RespiratoryDataset(csv_path=train_csv_tmp, feat_dir=FEAT_DIR)
val_ds = RespiratoryDataset(csv_path=val_csv_tmp, feat_dir=FEAT_DIR)

# 实例化 DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 3. 模型、损失函数、优化器
# ==========================================
model = build_model(in_dim=256, d_model=256, n_layers=4).to(device)

# 二分类交叉熵损失
criterion = nn.BCEWithLogitsLoss()

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# 混合精度加速器 (节省显存)
scaler = torch.cuda.amp.GradScaler()

# ==========================================
# 4. 训练与验证函数
# ==========================================

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for feats, labels in pbar:
        feats, labels = feats.to(device), labels.to(device).float()

        with torch.cuda.amp.autocast():
            file_logit, _ = model(feats)
            # 确保 file_logit 形状与 labels 一致 (B,)
            loss = criterion(file_logit.view(-1), labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # SE = Recall(positive)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall = se

    # SP = Specificity
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ICBHI score
    icbhi = 0.5 * (se + sp)

    # F1
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC（可能 NaN）
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float("nan")

    return {
        "AUC": auc,
        "F1": f1,
        "Recall": recall,
        "SE": se,
        "SP": sp,
        "ICBHI": icbhi,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }



# ==========================================
# 5. 主训练循环
# ==========================================
best_icbhi = -1
print("🚀 开始训练...")

try:
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(epoch)
        metrics = validate(thr=0.5)

        print(
            f"✅ Epoch {epoch} 总结: Loss: {avg_loss:.4f} | "
            f"ICBHI: {metrics['ICBHI']:.4f} | "
            f"SE: {metrics['SE']:.4f} | SP: {metrics['SP']:.4f} | "
            f"AUC: {metrics['AUC']:.4f} | F1: {metrics['F1']:.4f} | Recall: {metrics['Recall']:.4f} | "
            f"TP:{metrics['TP']} TN:{metrics['TN']} FP:{metrics['FP']} FN:{metrics['FN']}"
        )

        # ✅ 用 ICBHI 作为最优指标保存
        if metrics["ICBHI"] > best_icbhi:
            best_icbhi = metrics["ICBHI"]
            checkpoint_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🌟 新最优 ICBHI={best_icbhi:.4f}，已保存至: {checkpoint_path}")

finally:
    # 训练结束后清理临时文件（可选）
    if os.path.exists(train_csv_tmp): os.remove(train_csv_tmp)
    if os.path.exists(val_csv_tmp): os.remove(val_csv_tmp)

print("🏁 训练结束!")