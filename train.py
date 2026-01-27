# train_patient_split_vima.py
# ✅ Patient split + 全套指标(ACC/F1/AUC/SE/SP/ICBHI) + 混淆矩阵PNG
# ✅ 兼容你现在的 metadata.csv 列: ['wav_name','label','original_file']
# ✅ patient_id 从 original_file 里解析：默认取 '_' 前的第一段；解析不出就用完整文件名
# ✅ 直接可复制运行：python train_patient_split_vima.py

import os
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# 你的模型文件：确保 VimA_Model.py 在同目录，且里面有 VimAHybrid 类
from VimA_Model import VimAHybrid

# ================= 配置区 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR  = os.path.join(BASE_DIR, "spec_npy_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

# 如果你要调类权重：pos_weight = neg/pos（建议用 train 集统计）
USE_AUTO_POS_WEIGHT = True
MANUAL_POS_WEIGHT_VALUE = 1.0  # 当 USE_AUTO_POS_WEIGHT=False 时生效

# 阈值（也可以后面做 val 上的阈值搜索）
THRESH = 0.5

# patient_id 解析：优先从 original_file 抽取
# ICBHI 常见文件名：例如 101_1b1_Al_sc_Meditron.wav -> patient=101
PATIENT_PARSE_REGEX = r"^(\d+)"  # 开头的数字
FALLBACK_SPLIT_CHAR = "_"        # 再退一步，取 '_' 前一段

# 保存
BEST_CKPT_PATH = os.path.join(BASE_DIR, "best_vima_patient_split.pth")
CM_FINAL_PATH  = os.path.join(BASE_DIR, "confusion_matrix_final.png")
CM_BEST_PATH   = os.path.join(BASE_DIR, "confusion_matrix_best.png")


# ================= 工具函数 =================
def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0

def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = safe_div(tp, tp + fn)
    sp = safe_div(tn, tn + fp)
    icbhi = (se + sp) / 2.0

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "acc": acc, "f1": f1, "auc": auc,
        "se": se, "sp": sp, "icbhi": icbhi,
        "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def plot_and_save_cm(cm, path, title):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal(0)", "Abnormal(1)"],
        yticklabels=["Normal(0)", "Abnormal(1)"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def parse_patient_id(original_file: str) -> str:
    """
    从 original_file 解析 patient_id
    1) 正则：开头数字
    2) fallback：'_' 前一段
    3) 再 fallback：整个 original_file
    """
    if not isinstance(original_file, str) or len(original_file) == 0:
        return "unknown"

    m = re.match(PATIENT_PARSE_REGEX, original_file)
    if m:
        return m.group(1)

    if FALLBACK_SPLIT_CHAR in original_file:
        return original_file.split(FALLBACK_SPLIT_CHAR)[0]

    return original_file


# ================= Dataset =================
class ICBHIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, npy_dir: str):
        self.df = df.reset_index(drop=True)
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = os.path.join(self.npy_dir, row["wav_name"].replace(".wav", ".npy"))
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"找不到: {npy_path}")

        spec = np.load(npy_path)  # (128, T) 你现在是 (128,1024) 也没问题
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)  # [1, 128, T]
        label = torch.tensor(row["label"], dtype=torch.float)  # 0/1
        return spec_t, label


# ================= 主训练逻辑 =================
def main():
    print(f"Device: {DEVICE}")
    print("读取 metadata.csv ...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # 检查列
    need_cols = ["wav_name", "label", "original_file"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"metadata.csv 缺少列: {c}，当前列: {list(df.columns)}")

    # 生成 patient_id
    df["patient_id"] = df["original_file"].apply(parse_patient_id)

    # ========== Patient split（GroupShuffleSplit）==========
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))
    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(f"Train patients: {train_df['patient_id'].nunique()} | Val patients: {val_df['patient_id'].nunique()}")

    # 统计 label 分布
    print("\nLabel distribution:")
    print("Train:\n", train_df["label"].value_counts())
    print("Val:\n",   val_df["label"].value_counts())

    # DataLoader
    train_loader = DataLoader(
        ICBHIDataset(train_df, NPY_DIR),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        ICBHIDataset(val_df, NPY_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 模型
    print("初始化 VimA 模型 ...")
    model = VimAHybrid(num_classes=1, d_model=192, patch_time=1).to(DEVICE)

    # pos_weight（建议按 train 集统计）
    if USE_AUTO_POS_WEIGHT:
        pos = float((train_df["label"] == 1).sum())
        neg = float((train_df["label"] == 0).sum())
        pos_weight_value = (neg / pos) if pos > 0 else 1.0
    else:
        pos_weight_value = float(MANUAL_POS_WEIGHT_VALUE)

    print(f"pos_weight = {pos_weight_value:.4f} (neg/pos on train)")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=DEVICE)
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_icbhi = -1.0
    best_epoch = 0
    best_cm = None

    print("\n开始训练 ...")
    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for specs, labels in train_loader:
            specs = specs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)

        # ---- Val ----
        model.eval()
        all_labels, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits = model(specs)
                probs = torch.sigmoid(logits)

                preds = (probs > THRESH).long()

                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        m = compute_metrics(all_labels, all_preds, all_probs)

        print(f"\n--- Epoch [{epoch}/{EPOCHS}] ---")
        print(f"Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Validation: ACC: {m['acc']:.4f} | F1: {m['f1']:.4f} | AUC: {m['auc']:.4f}")
        print(f"            SE: {m['se']:.4f} | SP: {m['sp']:.4f} | ICBHI Score: {m['icbhi']:.4f}")
        print(f"Confusion Matrix:\n{m['cm']}")
        print(f"(TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']})")

        # 保存 best（按 ICBHI）
        if m["icbhi"] > best_icbhi:
            best_icbhi = m["icbhi"]
            best_epoch = epoch
            best_cm = m["cm"].copy()
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print(f"⭐ 发现更高 ICBHI Score ({best_icbhi:.4f})，已保存: {BEST_CKPT_PATH}")

            # 同时保存 best 的混淆矩阵图
            plot_and_save_cm(best_cm, CM_BEST_PATH, f"Best CM (Epoch {best_epoch}) | ICBHI={best_icbhi:.4f}")
            print(f"📊 已保存 best 混淆矩阵图: {CM_BEST_PATH}")

        # 最后一个 epoch 保存最终 CM
        if epoch == EPOCHS:
            plot_and_save_cm(m["cm"], CM_FINAL_PATH, f"Final CM (Epoch {epoch})")
            print(f"📊 已保存 final 混淆矩阵图: {CM_FINAL_PATH}")

    print("\n✅ 训练完成！")
    print(f"Best ICBHI Score: {best_icbhi:.4f} @ Epoch {best_epoch}")
    if best_cm is not None:
        print(f"Best Confusion Matrix:\n{best_cm}")


if __name__ == "__main__":
    main()
