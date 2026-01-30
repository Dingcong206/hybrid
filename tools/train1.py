# tools/train_v2.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 导入你的模块
from mymodels.model import build_model
from utils.metrics import segment_metrics, user_metrics


class CoswaraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            feat = np.load(row["feature_path"]).astype(np.float32)
            if feat.shape[0] == 97:
                feat = feat[1:, :]
            if feat.shape != (96, 1024):
                feat = np.resize(feat, (96, 1024)).astype(np.float32)
        except Exception:
            feat = np.zeros((96, 1024), dtype=np.float32)

        # 数据增强：训练时随机 Mask 掉一小段时序特征
        if self.is_train and np.random.rand() < 0.3:
            mask_len = np.random.randint(5, 15)
            start = np.random.randint(0, 96 - mask_len)
            feat[start: start + mask_len, :] = 0

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(feat), label


def train():
    CSV_PATH = "/data/dingcong/hybrid/Coswara-Data/coswara_hear_patches.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------
    # 超参数调整
    # -------------------
    BATCH_SIZE = 64
    EPOCHS = 60  # 增加总轮次
    MAX_LR = 8e-5  # 稍微调低峰值 LR，让收敛更稳
    WEIGHT_DECAY = 5e-4  # 稍微加大权重衰减
    DROPOUT = 0.3  # 提高 Dropout 防止过拟合
    PATIENCE = 15  # 容忍度调高，等待 OneCycle 的后期爆发
    CLIP_GRAD = 1.0

    METRIC_MODE = "f1_sp"
    MIN_SP = 0.65

    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH_AUC = str(OUT_DIR / "best_user_auc_v2.pth")
    SAVE_PATH_F1 = str(OUT_DIR / "best_user_f1_v2.pth")

    # -------------------
    # 加载数据
    # -------------------
    df = pd.read_csv(CSV_PATH)
    users = df["user_id"].unique()
    user_labels = df.groupby("user_id")["label"].max().reindex(users).values.astype(int)

    train_users, val_users = train_test_split(
        users, test_size=0.2, random_state=42, stratify=user_labels
    )
    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    train_loader = DataLoader(
        CoswaraDataset(train_df, is_train=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        CoswaraDataset(val_df, is_train=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 建议 d_model 设为 512 如果显存允许
    model = build_model(input_dim=1024, d_model=512, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=WEIGHT_DECAY)

    counts = np.bincount(train_df["label"].values.astype(int))
    # 稍微压低权重，从 2.08 降到 1.5 左右，平衡 SP
    pos_weight = torch.tensor([1.5], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 延长热身期
        div_factor=10.0,
        final_div_factor=100.0
    )

    best_user_auc = -1.0
    best_user_f1 = -1.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feats).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # -------------------
        # 验证
        # -------------------
        model.eval()
        seg_probs = []
        with torch.no_grad():
            for feats, _ in val_loader:
                logits = model(feats.to(DEVICE)).view(-1)
                seg_probs.extend(torch.sigmoid(logits).cpu().numpy())
        seg_probs = np.asarray(seg_probs)

        u_auc, u_best = user_metrics(val_df, seg_probs, mode=METRIC_MODE, min_sp=MIN_SP)

        print(f"\n Epoch {epoch} User-Level: AUC={u_auc:.4f}, F1={u_best['f1']:.4f}, SP={u_best['sp']:.4f}")
        print(f"   Confusion Matrix:\n{u_best['cm']}")
        # -------------------
        # 保存逻辑
        # -------------------
        improved_f1 = False
        if u_best["f1"] > best_user_f1:
            best_user_f1 = u_best["f1"]
            torch.save(model.state_dict(), SAVE_PATH_F1)
            improved_f1 = True
            no_improve = 0  # 只有 F1 进步才重置早停
            print(f"New Best F1: {best_user_f1:.4f} -> Saved to {SAVE_PATH_F1}")
        else:
            no_improve += 1

        if u_auc > best_user_auc:
            best_user_auc = u_auc
            torch.save(model.state_dict(), SAVE_PATH_AUC)
            print(f" New Best AUC: {best_user_auc:.4f} -> Saved to {SAVE_PATH_AUC}")

        if no_improve >= PATIENCE:
            print(f"Early Stopping at Epoch {epoch}. Best F1: {best_user_f1:.4f}")
            break

if __name__ == "__main__":
    train()