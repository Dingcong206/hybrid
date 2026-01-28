import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

from SSA_Model import SSA_Model


# =========================
# Dataset：segment-level（每个segment一个样本）
# =========================
class SegDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["feature_path"]).astype(np.float32)  # (T, D) e.g. (97,1024)
        label = float(row["label"])
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.float32)


# =========================
# 阈值搜索：按 F1 / ACC / ICBHI 选最佳阈值
# =========================
def search_best_threshold(y_true, y_prob, metric="f1"):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).reshape(-1)

    best = {"thr": 0.5, "score": -1.0}

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
                "cm": cm,
            }

    # AUC 不依赖阈值（只要两类都存在）
    if len(np.unique(y_true)) > 1:
        best["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        best["auc"] = float("nan")
    return best


def main():
    # ===== 配置区 =====
    CSV_PATH = "/data/dingcong/hybrid/metadata_segmented.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    BEST_BY = "f1"  # "f1" / "acc" / "icbhi"
    SAVE_PATH = f"best_ssa_recording_split_{BEST_BY}.pth"

    # ===== 读 CSV =====
    df = pd.read_csv(CSV_PATH)

    required_cols = {"feature_path", "label", "original_wav"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"CSV 缺少列: {miss}，当前 columns={df.columns.tolist()}")

    print("✅ CSV columns =", df.columns.tolist())
    print("📊 label dist:\n", df["label"].value_counts())
    print("📈 label ratio:\n", df["label"].value_counts(normalize=True))

    # ===== 方案二：按 original_wav 划分 =====
    recs = df["original_wav"].unique()

    # 尽量做“录音级别的分层划分”：每条录音用其segment的max(label)当录音标签
    rec_label = df.groupby("original_wav")["label"].max().reindex(recs).values
    stratify = rec_label if len(np.unique(rec_label)) > 1 else None

    train_r, val_r = train_test_split(
        recs,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    train_df = df[df["original_wav"].isin(train_r)].copy()
    val_df = df[df["original_wav"].isin(val_r)].copy()

    print(f"\n✅ Recording split done")
    print(f"Train recordings={len(train_r)} | Val recordings={len(val_r)}")
    print(f"Train segments={len(train_df)} | Val segments={len(val_df)}")

    train_loader = DataLoader(
        SegDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        SegDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ===== 模型 =====
    model = SSA_Model(input_dim=1024, d_model=256, n_layers=6).to(DEVICE)

    # 你的 segment-level label 近似平衡：默认不加 pos_weight
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        # ---------- train ----------
        model.train()
        total_loss = 0.0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE).view(-1)

            optimizer.zero_grad()
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        # ---------- val (segment-level) ----------
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for feats, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                feats = feats.to(DEVICE)
                logits = model(feats).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().astype(int).tolist())

        best = search_best_threshold(all_labels, all_probs, metric=BEST_BY)

        print(
            f"\nEpoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"[SEG|recording_split] AUC: {best['auc']:.4f} | "
            f"ACC: {best['acc']:.4f} | F1: {best['f1']:.4f} | "
            f"SE: {best['se']:.4f} | SP: {best['sp']:.4f} | "
            f"ICBHI: {best['icbhi']:.4f} | Thr*: {best['thr']:.3f}"
        )
        print("Confusion Matrix (segment-level):\n", best["cm"])

        if best["score"] > best_score:
            best_score = best["score"]
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ New Best Saved by {BEST_BY} (score={best_score:.4f}) -> {SAVE_PATH}")

    print(f"\n✅ Done. Best {BEST_BY} = {best_score:.4f}. Saved at: {SAVE_PATH}")


if __name__ == "__main__":
    main()
