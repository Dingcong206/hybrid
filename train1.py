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


class SegDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(row["feature_path"]).astype(np.float32)  # (T,D) e.g. (97,1024)
        label = float(row["label"])
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.float32)


def search_best_threshold(y_true, y_prob, metric="icbhi", thr_low=0.05, thr_high=0.95, steps=181):
    """
    现在模型不再“全判1”，就把阈值范围放开，避免错过真正的最优阈值。
    metric: "icbhi" / "f1" / "acc"
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).reshape(-1)

    best = {"thr": 0.5, "score": -1.0}

    for thr in np.linspace(thr_low, thr_high, steps):
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

    best["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return best


def main():
    CSV_PATH = "/data/dingcong/hybrid/metadata_segmented.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    EPOCHS = 80                 # 你说 27 epoch 最优 -> 我把上限拉高，但会早停
    LR = 5e-5                   # ✅ ICBHI 任务更稳的学习率（比 1e-4 更不容易震荡）
    WEIGHT_DECAY = 1e-2

    BEST_BY = "icbhi"
    SAVE_PATH = f"best_ssa_recording_split_{BEST_BY}.pth"

    # ✅ 阈值搜索放开（你现在已经不会全判1了）
    THR_LOW = 0.05
    THR_HIGH = 0.95
    THR_STEPS = 181

    # ✅ 早停（ICBHI 长时间不提升就停）
    PATIENCE = 12
    MIN_DELTA = 1e-4

    df = pd.read_csv(CSV_PATH)
    required_cols = {"feature_path", "label", "original_wav"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"CSV 缺少列: {miss}，当前 columns={df.columns.tolist()}")

    print("✅ CSV columns =", df.columns.tolist())
    print("📊 전체 label dist:\n", df["label"].value_counts())
    print("📈 전체 label ratio:\n", df["label"].value_counts(normalize=True))

    # recording split（方案二）
    recs = df["original_wav"].unique()
    rec_label = df.groupby("original_wav")["label"].max().reindex(recs).values
    stratify = rec_label if len(np.unique(rec_label)) > 1 else None

    train_r, val_r = train_test_split(
        recs, test_size=0.2, random_state=42, stratify=stratify
    )
    train_df = df[df["original_wav"].isin(train_r)].copy()
    val_df = df[df["original_wav"].isin(val_r)].copy()

    print(f"\n✅ Recording split done")
    print(f"Train recordings={len(train_r)} | Val recordings={len(val_r)}")
    print(f"Train segments={len(train_df)} | Val segments={len(val_df)}")
    print("\n📊 Train label dist:\n", train_df["label"].value_counts())
    print("📊 Val label dist:\n", val_df["label"].value_counts())

    train_loader = DataLoader(SegDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(SegDataset(val_df), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = SSA_Model(input_dim=1024, d_model=256, n_layers=6).to(DEVICE)

    # ✅ 如果你之前出现“全判1”，千万别加 pos_weight
    # 现在你已经均衡了，也先不加，保持稳定
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = -1.0
    best_epoch = -1
    best_thr = 0.5
    best_cm = None

    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
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

        # val
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for feats, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                feats = feats.to(DEVICE)
                logits = model(feats).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().astype(int).tolist())

        best = search_best_threshold(
            all_labels, all_probs, metric=BEST_BY,
            thr_low=THR_LOW, thr_high=THR_HIGH, steps=THR_STEPS
        )

        y_pred_best = (np.array(all_probs) >= best["thr"]).astype(int)
        pred_pos_ratio = float(y_pred_best.mean())

        print(
            f"\nEpoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"AUC: {best['auc']:.4f} | ACC: {best['acc']:.4f} | F1: {best['f1']:.4f} | "
            f"SE: {best['se']:.4f} | SP: {best['sp']:.4f} | "
            f"ICBHI: {best['icbhi']:.4f} | Thr*: {best['thr']:.3f} | Pred1%: {pred_pos_ratio*100:.1f}%"
        )
        print("Confusion Matrix (segment-level):\n", best["cm"])

        # save best + early stop
        if best["score"] > best_score + MIN_DELTA:
            best_score = best["score"]
            best_epoch = epoch
            best_thr = best["thr"]
            best_cm = best["cm"]

            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ New Best Saved by {BEST_BY} (score={best_score:.4f}) -> {SAVE_PATH}")

            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"\n Early Stop: {PATIENCE} epochs no improvement. Best at epoch {best_epoch}.")
            break

    print(f"\n✅ Done. Best {BEST_BY} = {best_score:.4f} at epoch {best_epoch}")
    print(f"✅ Best Thr* = {best_thr:.3f}")
    if best_cm is not None:
        print("✅ Best Confusion Matrix:\n", best_cm)


if __name__ == "__main__":
    main()
