import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mymodels.model import build_model



# =====================================================
# 1) Dataset：读取 HeAR Patch 特征
# =====================================================
class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["feature_path"]).astype(np.float32)
            # HeAR: 97 tokens -> 去掉 CLS 留 96
            if feat.shape[0] == 97:
                feat = feat[1:, :]
            # 容错：确保 (96, 1024)
            if feat.shape != (96, 1024):
                feat = np.resize(feat, (96, 1024)).astype(np.float32)
        except Exception:
            feat = np.zeros((96, 1024), dtype=np.float32)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


# =====================================================
# 2) 通用：扫描阈值最大化 F1（返回 AUC/F1/Acc/SE/SP/CM/Thr）
# =====================================================
def metrics_from_probs(y_true, y_prob, thr_mode="f1"):
    """
    thr_mode="f1": 扫描阈值，取 F1 最大的阈值
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # AUC 需要至少包含两个类别
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    best = {"f1": -1.0, "thr": 0.5, "acc": 0.0, "se": 0.0, "sp": 0.0, "cm": None}
    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall/sensitivity
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        if f1 > best["f1"]:
            best.update({"f1": f1, "thr": float(thr), "acc": acc, "se": se, "sp": sp, "cm": cm})

    if best["cm"] is None:
        # 极端情况下兜底
        thr = 0.5
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        best.update({"f1": f1, "thr": float(thr), "acc": acc, "se": se, "sp": sp, "cm": cm})

    return auc, best["f1"], best["acc"], best["thr"], best["se"], best["sp"], best["cm"]


# =====================================================
# 3) Segment-level 指标：直接用每一行样本
# =====================================================
def get_segment_metrics(val_df, seg_probs):
    y_true = val_df["label"].values.astype(int)
    y_prob = np.asarray(seg_probs).astype(float)
    return metrics_from_probs(y_true, y_prob)


# =====================================================
# 4) User-level 指标：按 user_id 聚合后再算
# =====================================================
def get_user_metrics(val_df, seg_probs, agg="mean"):
    temp = val_df.copy()
    temp["prob"] = np.asarray(seg_probs).astype(float)

    if agg == "mean":
        user_res = temp.groupby("user_id").agg({"prob": "mean", "label": "max"}).reset_index()
    elif agg == "max":
        user_res = temp.groupby("user_id").agg({"prob": "max", "label": "max"}).reset_index()
    else:
        raise ValueError("agg must be 'mean' or 'max'")

    y_true = user_res["label"].values.astype(int)
    y_prob = user_res["prob"].values.astype(float)
    return metrics_from_probs(y_true, y_prob)


# =====================================================
# 5) Train：AdamW + OneCycleLR + 同时输出 segment/user
# =====================================================
def train():
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 超参（OneCycle 建议 max_lr 比你之前略大）----
    BATCH_SIZE = 64
    EPOCHS = 30               # OneCycle 通常 20~40 足够
    MAX_LR = 2e-4             # OneCycle 的峰值 lr（可试 1e-4 / 2e-4 / 3e-4）
    WEIGHT_DECAY = 3e-4
    DROPOUT = 0.15
    PATIENCE = 8              # 早停按 user-F1
    CLIP_GRAD = 1.0

    SAVE_PATH_AUC = "best_user_auc_onecycle.pth"
    SAVE_PATH_F1  = "best_user_f1_onecycle.pth"

    # ---- 读数据：user-level stratify 划分 ----
    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )
    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df   = df[df["user_id"].isin(val_users)].copy()

    # 打印段级分布
    train_labels = train_df["label"].values.astype(int)
    counts = np.bincount(train_labels)
    print("Train label distribution (segment-level):", counts, "| ratio 0:1 =", f"{counts[0]/max(counts[1],1):.2f}:1")

    # ---- DataLoader：F1 最大化更推荐 shuffle，不用 sampler ----
    train_ds = CoswaraDataset(train_df)
    val_ds   = CoswaraDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ---- 模型/优化器/loss ----
    model = build_model(
        input_dim=1024,
        d_model=256,
        dropout=0.15
    ).to(device)

    #model = SSA_Model(input_dim=1024, d_model=256, dropout=DROPOUT).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    # F1 最大化：BCE 通常比 focal 更稳、更好校准阈值
    criterion = nn.BCEWithLogitsLoss()

    # ---- OneCycleLR：每 step 调一次 ----
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,         # 10% warmup
        div_factor=10.0,       # 初始lr = max_lr/10
        final_div_factor=50.0  # 最终lr = max_lr/50
    )

    best_user_auc = -1.0
    best_user_f1  = -1.0
    early_stop_counter = 0

    print(f"🚀 Start training (AdamW + OneCycleLR, F1-max eval)")
    print(f"   train={len(train_df)} segs, val={len(val_df)} segs | epochs={EPOCHS}, max_lr={MAX_LR}")

    for epoch in range(1, EPOCHS + 1):
        # =======================
        # Train
        # =======================
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()

            if CLIP_GRAD is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()
            scheduler.step()  # ✅ OneCycleLR 每 step 调一次

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        # =======================
        # Validate: 收集 segment 概率
        # =======================
        model.eval()
        seg_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                seg_probs.extend(torch.sigmoid(logits).cpu().numpy())

        # 1) Segment-level metrics
        s_auc, s_f1, s_acc, s_thr, s_se, s_sp, s_cm = get_segment_metrics(val_df.reset_index(drop=True), seg_probs)

        # 2) User-level metrics（默认 mean 聚合；你也可以试 agg="max"）
        u_auc, u_f1, u_acc, u_thr, u_se, u_sp, u_cm = get_user_metrics(val_df.reset_index(drop=True), seg_probs, agg="mean")

        print(f"\n📌 [Epoch {epoch}] Segment-Level (thr@F1-max):")
        print(f"   AUC: {s_auc:.4f} | F1: {s_f1:.4f} | ACC: {s_acc:.4f} | SE: {s_se:.4f} | SP: {s_sp:.4f} | Thr: {s_thr:.2f}")
        print(f"   CM:\n{s_cm}")

        print(f"\n📈 [Epoch {epoch}] User-Level (mean agg, thr@F1-max):")
        print(f"   AUC: {u_auc:.4f} | F1: {u_f1:.4f} | ACC: {u_acc:.4f} | SE: {u_se:.4f} | SP: {u_sp:.4f} | Thr: {u_thr:.2f}")
        print(f"   CM:\n{u_cm}")

        # =======================
        # Save best models (按 user-level)
        # =======================
        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH_AUC)
            print(f"💾 best USER-AUC updated -> {best_user_auc:.4f} | saved: {SAVE_PATH_AUC}")

        if u_f1 > best_user_f1:
            best_user_f1 = u_f1
            torch.save(model.state_dict(), SAVE_PATH_F1)
            print(f"💾 best USER-F1 updated  -> {best_user_f1:.4f} | saved: {SAVE_PATH_F1}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            print(f"  Early stop: {PATIENCE} epochs no USER-F1 improvement. best USER-AUC={best_user_auc:.4f}, best USER-F1={best_user_f1:.4f}")
            break


if __name__ == "__main__":
    train()
