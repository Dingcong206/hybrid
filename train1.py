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
        feat = np.load(row["feature_path"]).astype(np.float32)

        # HeAR 输出 97 token: (1 CLS + 96 patches)
        if feat.shape[0] == 97:
            feat = feat[1:, :]  # (96, 1024)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 3) 基础指标（给定阈值）
# =====================================================
def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob > thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return se, sp, acc, auc, f1, cm


# =====================================================
# 4) 扫阈值找 best F1（核心修改）
# =====================================================
def find_best_f1_threshold(y_true, y_prob, thresholds=None):
    """
    返回：
    best_thr, best_f1, best_se, best_sp, best_acc, auc, best_cm
    """
    if thresholds is None:
        # 比较细一点就用 0.01 步长；想快点可改成 0.02/0.05
        thresholds = np.linspace(0.0, 1.0, 101)

    best = {
        "thr": 0.5,
        "f1": -1.0,
        "se": 0.0,
        "sp": 0.0,
        "acc": 0.0,
        "auc": 0.5,
        "cm": None
    }

    # AUC 与阈值无关，先算一次
    y_true_np = np.asarray(y_true).astype(int).reshape(-1)
    y_prob_np = np.asarray(y_prob).reshape(-1)
    try:
        auc = roc_auc_score(y_true_np, y_prob_np) if len(np.unique(y_true_np)) > 1 else 0.5
    except Exception:
        auc = 0.5

    for thr in thresholds:
        se, sp, acc, _, f1, cm = compute_metrics(y_true_np, y_prob_np, thr=float(thr))

        # 选 best F1；若 F1 相同，优先 SE 更高（筛查更友好），再优先 SP
        if (f1 > best["f1"]) or (np.isclose(f1, best["f1"]) and se > best["se"]) or \
           (np.isclose(f1, best["f1"]) and np.isclose(se, best["se"]) and sp > best["sp"]):
            best.update({
                "thr": float(thr),
                "f1": float(f1),
                "se": float(se),
                "sp": float(sp),
                "acc": float(acc),
                "auc": float(auc),
                "cm": cm
            })

    return best["thr"], best["f1"], best["se"], best["sp"], best["acc"], best["auc"], best["cm"]


# =====================================================
# 5) 训练主程序 (User-level Split)
# =====================================================
def train():
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 100
    WEIGHT_DECAY = 1e-5

    SAVE_PATH_BESTF1 = "best_hear_ssa_coswara_bestf1.pth"
    SAVE_PATH_BESTAUC = "best_hear_ssa_coswara_bestaUc.pth"  # 可删

    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()

    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

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

    train_ds = CoswaraDataset(train_df)
    val_ds = CoswaraDataset(val_df)

    train_labels = train_df["label"].values.astype(int)
    counts = np.bincount(train_labels, minlength=2)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SSA_Model(input_dim=1024, d_model=256).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = -1.0
    best_auc = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            feats = feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # -----------------
        # 验证：收集 prob
        # -----------------
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(DEVICE, non_blocking=True)
                logits = model(feats).view(-1)
                probs = torch.sigmoid(logits).detach().cpu().numpy()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

        # ✅ 扫阈值得到 best F1
        best_thr, epoch_best_f1, se, sp, acc, auc, cm = find_best_f1_threshold(all_labels, all_probs)

        print(f"\n📊 [Epoch {epoch}] Val Results (Best-F1 Threshold Search):")
        print(f"   AUC: {auc:.4f} | BestF1: {epoch_best_f1:.4f} | SE: {se:.4f} | SP: {sp:.4f} | ACC: {acc:.4f}")
        print(f"   Best Threshold: {best_thr:.2f}")
        print(f"   Confusion Matrix:\n{cm}")

        # ✅ 按 best F1 保存
        if epoch_best_f1 > best_f1:
            best_f1 = epoch_best_f1
            torch.save(model.state_dict(), SAVE_PATH_BESTF1)
            print(f"🏆 Best-F1 提升！已保存至 {SAVE_PATH_BESTF1} (F1={best_f1:.4f}, thr={best_thr:.2f})")

        # （可选）保留 best AUC 版本
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), SAVE_PATH_BESTAUC)
            print(f"⭐ Best-AUC 提升！已保存至 {SAVE_PATH_BESTAUC} (AUC={best_auc:.4f})")

        scheduler.step()


if __name__ == "__main__":
    train()
