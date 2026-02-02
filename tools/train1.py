import sys
import os


# 1. 获取当前脚本 (train1.py) 的绝对路径
current_script_path = os.path.abspath(__file__)

# 2. 找到项目的根目录 (也就是 PythonProject 这一级)
# 第一次 dirname 得到 tools/，第二次 dirname 得到 PythonProject/
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 3. 将根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)

# 4. 现在可以正确导入了（注意加上 mymodels 前缀）
from mymodels.model import build_model
from mymodels.dataset import RespiratoryDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import os
from tqdm import tqdm

# 导入你之前的定义
# 向上退一级再进入文件夹导入
# ==========================================
# 1. 基础配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 2  # 32k序列较长，建议先设为2
LR = 1e-4
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 数据准备
# ==========================================
# 假设你已经把 CSV 拆分成了 train.csv 和 val.csv
train_ds = RespiratoryDataset(csv_path="train.csv", feat_dir="/data/dingcong/hybrid/hear_16x256_fixed")
val_ds = RespiratoryDataset(csv_path="val.csv", feat_dir="/data/dingcong/hybrid/labels")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 3. 模型、损失函数、优化器
# ==========================================
model = build_model(in_dim=256, d_model=256, n_layers=4).to(device)

# 二分类交叉熵损失 (带 Logits 稳定性更好)
criterion = nn.BCEWithLogitsLoss()

# AdamW 是训练 Mamba/Transformer 类模型的首选
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# 混合精度加速器 (节省显存)
scaler = torch.cuda.amp.GradScaler()


# ==========================================
# 4. 训练与验证逻辑
# ==========================================

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for feats, labels in pbar:
        feats, labels = feats.to(device), labels.to(device).float()

        # 使用自动混合精度
        with torch.cuda.amp.autocast():
            # file_logit 是全文件的预测，patch_logits 是每个点的预测
            file_logit, _ = model(feats)
            loss = criterion(file_logit, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # 缩放梯度
        scaler.step(optimizer)  # 更新参数
        scaler.update()  # 更新缩放因子

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)


def validate():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in tqdm(val_loader, desc="[Valid]"):
            feats = feats.to(device)
            file_logit, _ = model(feats)

            # 记录预测概率 (Sigmoid 之后) 和 真实标签
            prob = torch.sigmoid(file_logit)
            all_preds.extend(prob.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算医学任务的核心指标：AUC
    auc = roc_auc_score(all_labels, all_preds)
    # 计算 F1 Score (需要把概率转成 0 或 1)
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    f1 = f1_score(all_labels, preds_binary)

    return auc, f1


# ==========================================
# 5. 主循环
# ==========================================
best_auc = 0
for epoch in range(1, EPOCHS + 1):
    avg_loss = train_one_epoch(epoch)
    val_auc, val_f1 = validate()

    print(f"--- Epoch {epoch} Results: Loss: {avg_loss:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} ---")

    # 只要 AUC 有提升就保存模型
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"🌟 New Best AUC! Model saved.")