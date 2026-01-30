import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 确保你的 SSA_Model.py 就在同级目录下
from SSA_Model import SSA_Model


# =====================================================
# 1) Dataset：适配 HeAR 特征读取
# =====================================================
class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 加载特征 (97, 1024)
        try:
            feat = np.load(row["feature_path"]).astype(np.float32)
            # HeAR 输出 97 个 token，去掉第一个 CLS 留 96 个给 SSA
            if feat.shape[0] == 97:
                feat = feat[1:, :]
            # 容错：保证形状是 (96, 1024)
            if feat.shape != (96, 1024):
                feat = np.resize(feat, (96, 1024)).astype(np.float32)
        except Exception:
            feat = np.zeros((96, 1024), dtype=np.float32)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 2) 损失函数：Focal Loss + 可选标签平滑
#   说明：你是 2:1 中度不平衡，不建议 gamma 过大
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.65, gamma=1.5, smoothing=0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 标签平滑（可选；二分类里一般别太大）
        if self.smoothing > 0:
            with torch.no_grad():
                targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# =====================================================
# 3) User-Level 指标计算：支持两种阈值策略
#    - mode="f1": 扫描阈值取 F1 最大
#    - mode="recall": 先满足 recall>=target_recall，再取 F1 最大（筛查更稳）
# =====================================================
def get_user_metrics(df, probs, mode="f1", target_recall=0.80):
    temp_df = df.copy()
    temp_df["prob"] = probs

    # user-level 聚合：prob 平均；label max（用户有任一阳性就算阳性）
    user_res = temp_df.groupby("user_id").agg({"prob": "mean", "label": "max"}).reset_index()

    y_true = user_res["label"].values.astype(int)
    y_prob = user_res["prob"].values.astype(float)

    auc = roc_auc_score(y_true, y_prob)

    best = {"f1": -1.0, "thr": 0.5, "se": 0.0, "sp": 0.0, "cm": None}

    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall/sensitivity
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if mode == "recall" and se < target_recall:
            continue

        if f1 > best["f1"]:
            best.update({"f1": f1, "thr": float(thr), "se": se, "sp": sp, "cm": cm})

    # 如果 recall 模式找不到满足 recall 的阈值，回退到 f1 模式
    if best["cm"] is None:
        return get_user_metrics(df, probs, mode="f1", target_recall=target_recall)

    return auc, best["f1"], best["thr"], best["se"], best["sp"], best["cm"]


# =====================================================
# 4) 训练主程序（已改：保存 best AUC + best F1；早停可选看 F1/混合；打印 recall）
# =====================================================
def train():
    # --- 路径与设备 ---
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 训练超参 ---
    BATCH_SIZE = 64
    LR = 8e-5
    EPOCHS = 50
    WEIGHT_DECAY = 5e-4
    PATIENCE = 6

    # --- 保存策略 ---
    SAVE_PATH_AUC = "best_ssa_user_auc.pth"
    SAVE_PATH_F1 = "best_ssa_user_f1.pth"

    # --- 评估/早停策略（按你的目标选）---
    THR_MODE = "recall"        # "f1" 或 "recall"
    TARGET_RECALL = 0.80       # recall 模式下的下限，建议 0.80~0.90
    EARLY_STOP_ON = "f1"       # "auc" / "f1" / "mix"
    MIX_W = 0.5                # mix 分数 = auc + MIX_W*f1

    # --- 1) 读数据 + user-level 分层划分 ---
    df = pd.read_csv(CSV_PATH)

    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    # --- 2) DataLoader + 采样 ---
    train_ds = CoswaraDataset(train_df)
    val_ds = CoswaraDataset(val_df)

    train_labels = train_df["label"].values.astype(int)
    counts = np.bincount(train_labels)
    print("Train label distribution:", counts, "| ratio 0:1 =", f"{counts[0]/max(counts[1],1):.2f}:1")

    # 中度不平衡：可以用 WeightedRandomSampler（你现在就用这个）
    weights = 1.0 / (counts + 1e-6)
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # --- 3) 初始化模型/优化器/损失/调度器 ---
    model = SSA_Model(input_dim=1024, d_model=256, dropout=0.3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 你是 2:1 中度不平衡：推荐 alpha 稍大、gamma 稍小
    criterion = FocalLoss(alpha=0.65, gamma=1.5, smoothing=0.0)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_auc = -1.0
    best_f1 = -1.0
    best_mix = -1e9
    early_stop_counter = 0

    print(f"🚀 开始训练! 训练样本: {len(train_df)}, 验证样本: {len(val_df)}")
    print(f"⚙ THR_MODE={THR_MODE} | TARGET_RECALL={TARGET_RECALL} | EARLY_STOP_ON={EARLY_STOP_ON}")

    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 稳一点（可删）
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # --- 验证：收集 val 概率 ---
        model.eval()
        all_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        # --- user-level 指标 ---
        u_auc, u_f1, u_thr, u_se, u_sp, u_cm = get_user_metrics(
            val_df, all_probs, mode=THR_MODE, target_recall=TARGET_RECALL
        )

        print(f"\n📈 [Epoch {epoch}] User-Level Result:")
        print(f"   AUC: {u_auc:.4f} | F1: {u_f1:.4f} | Recall(SE): {u_se:.4f} | SP: {u_sp:.4f} | Thr: {u_thr:.2f}")
        print(f"   Confusion Matrix:\n{u_cm}")

        # --- 保存 best AUC ---
        if u_auc > best_auc:
            best_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH_AUC)
            print(f"💾 best AUC updated -> {best_auc:.4f} | saved: {SAVE_PATH_AUC}")

        # --- 保存 best F1 ---
        if u_f1 > best_f1:
            best_f1 = u_f1
            torch.save(model.state_dict(), SAVE_PATH_F1)
            print(f"💾 best F1 updated  -> {best_f1:.4f} | saved: {SAVE_PATH_F1}")

        # --- Early stopping 依据 ---
        improved = False
        if EARLY_STOP_ON == "auc":
            improved = (u_auc >= best_auc)  # best_auc 已更新
        elif EARLY_STOP_ON == "f1":
            improved = (u_f1 >= best_f1)    # best_f1 已更新
        else:  # "mix"
            mix = u_auc + MIX_W * u_f1
            if mix > best_mix:
                best_mix = mix
                improved = True

        if improved:
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print(
                f"🛑 连续 {PATIENCE} 轮无提升，触发早停。"
                f" best AUC={best_auc:.4f}, best F1={best_f1:.4f}"
            )
            break

        scheduler.step()


if __name__ == "__main__":
    train()