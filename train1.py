import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

# ===== 引入你的模型 =====
from SSA_Model import SSA_Model


# =====================================================
# 1) Dataset：返回 feature, label, patient_id
# =====================================================
class ICBHIDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["feature_path"]).astype(np.float32)  # (97,1024) or (T,D)
        label = float(row["label"])
        pid = str(row["patient_id"])
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.float32), pid


# =====================================================
# 2) patient-level 聚合
# =====================================================
def aggregate_patient_probs(patient_ids, probs, labels, mode="mean"):
    """
    将 segment-level 概率聚合到 patient-level。
    mode:
      - mean: 病人所有 segment 概率平均
      - max : 取最大（更偏 SE）
      - vote: segment 用 0.5 投票，投票比例当作“概率”
    """
    patient_ids = np.array(patient_ids)
    probs = np.array(probs).reshape(-1)
    labels = np.array(labels).astype(int).reshape(-1)

    uniq = np.unique(patient_ids)
    y_true_p, y_prob_p = [], []

    for pid in uniq:
        idx = (patient_ids == pid)
        p_probs = probs[idx]
        p_labels = labels[idx]

        true_label = int(p_labels.max())  # 病人标签：只要有异常段，就算异常

        if mode == "mean":
            prob = float(p_probs.mean())
        elif mode == "max":
            prob = float(p_probs.max())
        elif mode == "vote":
            pred_seg = (p_probs >= 0.5).astype(int)
            prob = float(pred_seg.mean())
        else:
            raise ValueError("mode must be one of: mean/max/vote")

        y_true_p.append(true_label)
        y_prob_p.append(prob)

    return np.array(y_true_p), np.array(y_prob_p)


# =====================================================
# 3) 阈值搜索（按 icbhi / f1 / acc 选择最优阈值）
# =====================================================
def search_best_threshold(y_true, y_prob, metric="icbhi"):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).reshape(-1)

    best = {"thr": 0.5, "score": -1}

    for thr in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        icbhi = 0.5 * (se + sp)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        score = {"icbhi": icbhi, "f1": f1, "acc": acc}[metric]

        if score > best["score"]:
            best = {
                "thr": float(thr),
                "score": float(score),
                "acc": float(acc),
                "f1": float(f1),
                "se": float(se),
                "sp": float(sp),
                "icbhi": float(icbhi),
                "cm": cm
            }

    # AUC 不依赖阈值
    if len(np.unique(y_true)) > 1:
        best["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        best["auc"] = float("nan")
    return best


# =====================================================
# 4) 主训练
# =====================================================
def main():
    # --------- 配置区：按需改 ---------
    CSV_PATH = "/data/dingcong/hybrid/metadata_segmented.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # patient-level 聚合方式：mean / max / vote
    PATIENT_AGG = "mean"

    # 自动阈值选择依据：icbhi / f1 / acc
    #   - 医学任务推荐 icbhi
    #   - 你如果更想拉高整体识别，可用 f1
    BEST_BY = "icbhi"

    SAVE_PATH = f"best_ssa_{BEST_BY}_patient_level.pth"

    # early stopping（可选）
    USE_EARLY_STOP = True
    PATIENCE = 12
    # --------------------------------

    # 读数据
    df = pd.read_csv(CSV_PATH)
    print("✅ CSV_PATH =", CSV_PATH)
    print("✅ columns =", df.columns.tolist())

    # 生成 patient_id
    id_col = "original_wav" if "original_wav" in df.columns else "user_id"
    df["patient_id"] = df[id_col].apply(lambda x: str(x).split("_")[0])

    # patient-level split（80/20）
    unique_patients = df["patient_id"].unique()
    train_p, val_p = train_test_split(unique_patients, test_size=0.2, random_state=42)

    train_df = df[df["patient_id"].isin(train_p)].copy()
    val_df = df[df["patient_id"].isin(val_p)].copy()

    print(f"✅ patient split: train patients={len(train_p)} | val patients={len(val_p)}")
    print("📊 segment label dist (train):\n", train_df["label"].value_counts())
    print("📊 segment label dist (val):\n", val_df["label"].value_counts())

    train_loader = DataLoader(ICBHIDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ICBHIDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型（输入是 HeAR 的 embed_dim=1024）
    model = SSA_Model(input_dim=1024, d_model=256, n_layers=6).to(DEVICE)

    # 你的 segment-level 数据几乎平衡：默认不加 pos_weight
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = -1
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        # ----------------- train -----------------
        model.train()
        total_loss = 0.0

        for feats, labels, _pids in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            feats = feats.to(DEVICE)               # (B,97,1024)
            labels = labels.to(DEVICE).view(-1)    # (B,)

            optimizer.zero_grad()
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        # ----------------- val -----------------
        model.eval()
        all_probs, all_labels, all_pids = [], [], []

        with torch.no_grad():
            for feats, labels, pids in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                feats = feats.to(DEVICE)
                logits = model(feats).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().astype(int).tolist())
                all_pids.extend(list(pids))

        # patient-level 聚合
        y_true_p, y_prob_p = aggregate_patient_probs(all_pids, all_probs, all_labels, mode=PATIENT_AGG)

        # 自动阈值 + 指标
        best = search_best_threshold(y_true_p, y_prob_p, metric=BEST_BY)

        print(
            f"\nEpoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"[Patient-{PATIENT_AGG}] AUC: {best['auc']:.4f} | "
            f"ACC: {best['acc']:.4f} | F1: {best['f1']:.4f} | "
            f"SE: {best['se']:.4f} | SP: {best['sp']:.4f} | "
            f"ICBHI: {best['icbhi']:.4f} | Thr*: {best['thr']:.3f}"
        )
        print("Confusion Matrix (patient-level):\n", best["cm"])

        # 保存 best
        if best["score"] > best_score:
            best_score = best["score"]
            bad_epochs = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ New Best Saved by {BEST_BY} (score={best_score:.4f}) -> {SAVE_PATH}")
        else:
            bad_epochs += 1

        if USE_EARLY_STOP and bad_epochs >= PATIENCE:
            print(f" Early stop: {PATIENCE} epochs no improvement.")
            break

    print(f"\n✅ Done. Best {BEST_BY} = {best_score:.4f}. Model saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()
