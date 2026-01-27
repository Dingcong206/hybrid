import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

from SSA_Model import SSA_Model

# ================= 配置区（Coswara）=================
BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"
CSV_PATH = os.path.join(BASE_DIR, "metadata_multi.csv")
NPY_DIR = os.path.join(BASE_DIR, "coswara_multi_modal_npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.1
WARMUP_EPOCHS = 5

BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_ssa_coswara_by_f1.pth")
CM_BEST_PATH = os.path.join(BASE_DIR, "confusion_matrix_coswara_by_f1.png")


# ================= 数据增强 =================
def apply_spec_augment(spec, max_f=15, max_t=80):
    """
    spec: [F, T]，一般你的数据是 [128, 1024]
    """
    F, T = spec.shape

    if random.random() > 0.5:
        f = random.randint(5, min(max_f, F))
        f0 = random.randint(0, max(F - f, 1) - 1) if (F - f) > 0 else 0
        spec[f0:f0 + f, :] = 0

    if random.random() > 0.5:
        t = random.randint(20, min(max_t, T))
        t0 = random.randint(0, max(T - t, 1) - 1) if (T - t) > 0 else 0
        spec[:, t0:t0 + t] = 0

    return spec


# ================= Dataset（Coswara，多模态展开）=================
class CoswaraDataset(Dataset):
    """
    读取 metadata_multi.csv:
      - user_id: 用户ID
      - label: 0/1
      - modes: 逗号分隔，如 "cough-shallow,breathing-deep"
    对每个 user 的 modes 展开为多个 npy 样本：
      npy 文件名: {user_id}_{mode}.npy
    """
    def __init__(self, df, npy_dir, is_train=False):
        self.npy_dir = npy_dir
        self.is_train = is_train
        self.samples = []

        for _, row in df.iterrows():
            user_id = str(row["user_id"])
            label = int(row["label"])
            modes = str(row["modes"]).split(",")

            for mode in modes:
                mode = mode.strip()
                self.samples.append({
                    "npy_path": os.path.join(self.npy_dir, f"{user_id}_{mode}.npy"),
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        spec = np.load(item["npy_path"])  # [F, T]

        if self.is_train:
            spec = apply_spec_augment(spec)

        spec_t = torch.from_numpy(spec).float().unsqueeze(0)  # [1, F, T]
        label = torch.tensor(item["label"], dtype=torch.float32)  # scalar
        return spec_t, label


# ================= 阈值搜索：F1 最大 =================
def find_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray):
    """
    在验证集上搜索使 F1 最大的阈值
    为了更稳更快：用概率分位点作为候选阈值（201个）
    返回：best_thr, best_f1, best_cm, best_se, best_sp, best_acc
    """
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 201)))

    best_f1 = -1.0
    best_thr = 0.5
    best_cm = None
    best_se = best_sp = best_acc = 0.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_cm = cm
            best_se, best_sp, best_acc = float(se), float(sp), float(acc)

    return {
        "thr": best_thr,
        "f1": float(best_f1),
        "cm": best_cm,
        "se": best_se,
        "sp": best_sp,
        "acc": best_acc
    }


# ================= 主程序 =================
def main():
    df = pd.read_csv(CSV_PATH)

    # 1) 严格按 user_id 分割，防止数据泄露
    unique_users = df["user_id"].unique()
    train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    train_ds = CoswaraDataset(train_df, NPY_DIR, is_train=True)
    val_ds = CoswaraDataset(val_df, NPY_DIR, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2) 初始化 SSA_Model（你可以按需要调整 patch_time / n_layers）
    model = SSA_Model(num_classes=1, n_layers=4, d_model=192, patch_time=1).to(DEVICE)

    # 3) 类别不平衡：pos_weight = neg/pos（用 user-level df 统计，与之前一致）
    pos = float(train_df["label"].sum())
    neg = float(len(train_df) - train_df["label"].sum())
    if pos < 1:
        raise ValueError("训练集中正样本为 0，无法训练。请检查标签。")
    pw = torch.tensor([neg / pos], device=DEVICE, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 4) 学习率：预热 + 余弦退火（和你 trainSSA 版本一致）
    def lr_lambda(epoch_idx):
        # epoch_idx 从 0 开始
        if epoch_idx < WARMUP_EPOCHS:
            return float(epoch_idx + 1) / float(WARMUP_EPOCHS)
        # 余弦
        progress = (epoch_idx - WARMUP_EPOCHS) / max((EPOCHS - WARMUP_EPOCHS), 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_f1 = -1.0
    early_stop_counter = 0
    patience = 20

    print(f"🚀 SSA_Model | Coswara | 训练模态样本: {len(train_ds)} | 验证模态样本: {len(val_ds)}")
    print(f"📌 pos_weight = {pw.item():.4f} (neg/pos) | device={DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        # ================= 训练 =================
        model.train()
        train_loss = 0.0

        for specs, labels in train_loader:
            specs = specs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).view(-1)  # [B]

            optimizer.zero_grad()
            logits = model(specs).view(-1)  # [B]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / max(len(train_loader), 1)

        # ================= 验证 =================
        model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE, non_blocking=True)
                logits = model(specs).view(-1)  # [B]
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(labels.numpy())  # labels 在 CPU

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels).astype(int)

        # 指标：AUC
        auc = roc_auc_score(all_labels, all_probs)

        # 自动阈值：F1 最大
        metrics = find_best_threshold_by_f1(all_labels, all_probs)
        thr = metrics["thr"]
        f1v = metrics["f1"]
        acc = metrics["acc"]
        se = metrics["se"]
        sp = metrics["sp"]
        cm = metrics["cm"]

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"AUC: {auc:.4f} | ACC: {acc:.4f} | F1: {f1v:.4f} | "
            f"SE: {se:.4f} | SP: {sp:.4f} | Thr*: {thr:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step()

        # ================= best（按 F1）+ early stopping =================
        if f1v > best_f1:
            best_f1 = f1v
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print(f"⭐ New Best Model Saved (by F1={best_f1:.4f})")

            # 保存混淆矩阵图
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["True 0", "True 1"])
            plt.title(f"Best Confusion Matrix (F1={best_f1:.4f}, Thr={thr:.3f})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(CM_BEST_PATH, dpi=200)
            plt.close()
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"连续 {patience} 轮 F1 未提升，提前停止。")
            break

    print(f"✅ 训练完成! 历史最高验证集 F1: {best_f1:.4f}")
    print(f"✅ Best checkpoint: {BEST_CKPT_PATH}")
    print(f"✅ Best confusion matrix: {CM_BEST_PATH}")


if __name__ == "__main__":
    main()
