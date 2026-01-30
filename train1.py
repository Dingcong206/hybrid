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
# 1) Focal Loss
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
# 2) Dataset：返回 user_id（新增）
# =====================================================
class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["feature_path"]).astype(np.float32)

        if feat.shape[0] == 97:
            feat = feat[1:, :]  # (96, 1024)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        user_id = row["user_id"]
        return torch.from_numpy(feat), label, user_id


# =====================================================
# 3) 指标
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


def find_best_f1_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    best = {"thr": 0.5, "f1": -1.0, "se": 0.0, "sp": 0.0, "acc": 0.0, "auc": 0.5, "cm": None}

    y_true_np = np.asarray(y_true).astype(int).reshape(-1)
    y_prob_np = np.asarray(y_prob).reshape(-1)

    try:
        auc = roc_auc_score(y_true_np, y_prob_np) if len(np.unique(y_true_np)) > 1 else 0.5
    except Exception:
        auc = 0.5

    for thr in thresholds:
        se, sp, acc, _, f1, cm = compute_metrics(y_true_np, y_prob_np, thr=float(thr))
        if (f1 > best["f1"]) or (np.isclose(f1, best["f1"]) and se > best["se"]) or \
           (np.isclose(f1, best["f1"]) and np.isclose(se, best["se"]) and sp > best["sp"]):
            best.update({"thr": float(thr), "f1": float(f1), "se": float(se), "sp": float(sp),
                         "acc": float(acc), "auc": float(auc), "cm": cm})

    return best["thr"], best["f1"], best["se"], best["sp"], best["acc"], best["auc"], best["cm"]


# =====================================================
# 4) 用户级聚合（新增）
# =====================================================
def aggregate_user_probs(user_ids, y_true, y_prob, mode="mean"):
    """
    将同一 user 的多个样本概率聚合成一个概率，然后按 user 评估
    mode: "mean" or "max"
    返回: user_true, user_prob
    """
    df_tmp = pd.DataFrame({
        "user_id": user_ids,
        "y_true": np.asarray(y_true).astype(int),
        "y_prob": np.asarray(y_prob).astype(float)
    })

    # user 的标签取 max（只要该用户有一次阳性，就算阳性）
    user_true = df_tmp.groupby("user_id")["y_true"].max()

    if mode == "max":
        user_prob = df_tmp.groupby("user_id")["y_prob"].max()
    else:
        user_prob = df_tmp.groupby("user_id")["y_prob"].mean()

    return user_true.values, user_prob.values


# =====================================================
# 5) Train
# =====================================================
def train():
    BASE_DIR = "/data/dingcong/hybrid/Coswara-Data"  # 你数据所在目录
    CSV_PATH = os.path.join(BASE_DIR, "coswara_hear_patches_yamnet.csv")  # 注意是 yamnet
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    LR = 1e-4
    EPOCHS = 100
    WEIGHT_DECAY = 1e-5

    # ✅ 改：按“用户级 bestF1”保存更符合你的任务
    SAVE_PATH_BESTF1_USER = "best_hear_ssa_coswara_user_bestf1.pth"
    SAVE_PATH_BESTAUC_USER = "best_hear_ssa_coswara_user_bestaUc.pth"

    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    print("✅ 数据划分完成:")
    print(f"   训练集: {len(train_users)} 用户, {len(train_df)} 样本")
    print(f"   验证集: {len(val_users)} 用户, {len(val_df)} 样本")

    train_ds = CoswaraDataset(train_df)
    val_ds = CoswaraDataset(val_df)

    # 类别不平衡：Sampler
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

    best_user_f1 = -1.0
    best_user_auc = -1.0

    for epoch in range(1, EPOCHS + 1):
        # -------- Train --------
        model.train()
        for feats, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            feats = feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # -------- Val: collect sample probs + user ids --------
        model.eval()
        all_labels, all_probs, all_uids = [], [], []
        with torch.no_grad():
            for feats, labels, uids in val_loader:
                feats = feats.to(DEVICE, non_blocking=True)
                logits = model(feats).view(-1)
                probs = torch.sigmoid(logits).detach().cpu().numpy()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())
                all_uids.extend(list(uids))

        # (1) sample-level bestF1（保留输出，方便你看）
        s_thr, s_f1, s_se, s_sp, s_acc, s_auc, s_cm = find_best_f1_threshold(all_labels, all_probs)

        # (2) user-level (mean) bestF1（✅ 主要看这个）
        user_y, user_p = aggregate_user_probs(all_uids, all_labels, all_probs, mode="mean")
        u_thr, u_f1, u_se, u_sp, u_acc, u_auc, u_cm = find_best_f1_threshold(user_y, user_p)

        print(f"\n📊 [Epoch {epoch}] Val Results")
        print(f"   [Sample] AUC: {s_auc:.4f} | BestF1: {s_f1:.4f} | SE: {s_se:.4f} | SP: {s_sp:.4f} | Thr: {s_thr:.2f}")
        print(f"   [Sample] CM:\n{s_cm}")

        print(f"   [UserMean] AUC: {u_auc:.4f} | BestF1: {u_f1:.4f} | SE: {u_se:.4f} | SP: {u_sp:.4f} | Thr: {u_thr:.2f}")
        print(f"   [UserMean] CM:\n{u_cm}")

        # ✅ 按 user-level best F1 保存
        if u_f1 > best_user_f1:
            best_user_f1 = u_f1
            torch.save(model.state_dict(), SAVE_PATH_BESTF1_USER)
            print(f"🏆 UserBest-F1 提升！已保存至 {SAVE_PATH_BESTF1_USER} (F1={best_user_f1:.4f}, thr={u_thr:.2f})")

        # ✅ 按 user-level best AUC 保存（可选）
        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH_BESTAUC_USER)
            print(f"⭐ UserBest-AUC 提升！已保存至 {SAVE_PATH_BESTAUC_USER} (AUC={best_user_auc:.4f})")

        scheduler.step()


if __name__ == "__main__":
    train()
