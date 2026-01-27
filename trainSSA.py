import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 导入你刚才命名的 SSA_Model
from SSA_Model import SSA_Model

# ================= 配置区 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-5  # SSA模型较深，建议学习率稍小
WEIGHT_DECAY = 0.1  # 增强权重衰减
WARMUP_EPOCHS = 5  # 预热轮数

PATIENT_PARSE_REGEX = r"^(\d+)"
BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_ssa_model.pth")
CM_BEST_PATH = os.path.join(BASE_DIR, "confusion_matrix_ssa.png")


# ================= 数据增强 =================
def apply_spec_augment(spec, max_f=15, max_t=80):
    """ 针对 SSA 优化的增强：稍微加大时间掩码范围 """
    if random.random() > 0.5:
        f = random.randint(5, max_f)
        f0 = random.randint(0, 128 - f)
        spec[f0:f0 + f, :] = 0
    if random.random() > 0.5:
        t = random.randint(20, max_t)
        t0 = random.randint(0, 1024 - t)
        spec[:, t0:t0 + t] = 0
    return spec


# ================= Dataset =================
class ICBHIDataset(Dataset):
    def __init__(self, df, npy_dir, is_train=False):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_name = str(row["wav_name"]).replace(".wav", ".npy")
        spec = np.load(os.path.join(self.npy_dir, npy_name))

        if self.is_train:
            spec = apply_spec_augment(spec)

        spec_t = torch.from_numpy(spec).float().unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.float)
        return spec_t, label


# ================= 阈值搜索工具 =================
def find_best_icbhi(y_true, y_prob):
    best_icbhi = -1
    best_metrics = {}
    # 在 0.1 到 0.9 之间搜索最佳分类阈值
    for thr in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        icbhi = (se + sp) / 2
        if icbhi > best_icbhi:
            best_icbhi = icbhi
            best_metrics = {"se": se, "sp": sp, "icbhi": icbhi, "thr": thr, "cm": cm}
    return best_metrics


# ================= 主程序 =================
def main():
    df = pd.read_csv(CSV_PATH)
    df["patient_id"] = df["original_file"].apply(
        lambda x: re.match(PATIENT_PARSE_REGEX, str(x)).group(1) if re.match(PATIENT_PARSE_REGEX,
                                                                             str(x)) else "unknown")

    # 按病人划分，确保验证集里没有训练集的病人
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR, is_train=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR, is_train=False),
                            batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # 初始化 SSA_Model
    model = SSA_Model(num_classes=1, n_layers=4, d_model=192, patch_time=4).to(DEVICE)

    # 自动计算正样本权重 (处理类别不平衡)
    pos = (train_df["label"] == 1).sum()
    neg = (train_df["label"] == 0).sum()
    pw = torch.tensor([neg / (pos + 1e-8)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 学习率调度：预热 + 余弦退火
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_icbhi = 0
    early_stop_counter = 0
    patience = 20

    print(f"🚀 SSA_Model 启动 | 训练样本: {len(train_df)} | 验证样本: {len(val_df)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE).view(-1)
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证逻辑
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE)
                logits = model(specs)
                all_labels.extend(labels.numpy())
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        # 搜索最佳阈值对应的 ICBHI
        metrics = find_best_icbhi(np.array(all_labels), np.array(all_probs))
        auc = roc_auc_score(all_labels, all_probs)

        print(f"Epoch [{epoch}/{EPOCHS}] Loss: {train_loss / len(train_loader):.4f} | "
              f"ICBHI: {metrics['icbhi']:.4f} | SE: {metrics['se']:.4f} | SP: {metrics['sp']:.4f} | AUC: {auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # 保存最优模型
        if metrics['icbhi'] > best_icbhi:
            best_icbhi = metrics['icbhi']
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_CKPT_PATH)

            # 绘制混淆矩阵
            plt.figure(figsize=(6, 5))
            sns.heatmap(metrics['cm'], annot=True, fmt="d", cmap="Greens")
            plt.title(f"SSA Best ICBHI: {best_icbhi:.4f}")
            plt.savefig(CM_BEST_PATH)
            plt.close()
            print(f"⭐ 新纪录！最优 ICBHI 已更新为 {best_icbhi:.4f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"🛑 连续 {patience} 轮未提升，提前停止。")
            break

    print(f"✅ 训练完成! 历史最高 ICBHI: {best_icbhi:.4f}")


if __name__ == "__main__":
    main()