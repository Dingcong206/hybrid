#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels import build_model

# ============================================================
# 0) 导入你的模型（model.py 里已提供 SSA_Model）
# ============================================================
from mymodels.model import SSA_Model


# ============================================================
# 1) 可选：SpecAugment（对 fbank: T x F 遮挡）
# ============================================================
def apply_spec_augment(x, max_mask_t=40, max_mask_f=16, num_masks=2):
    """
    x: torch.Tensor (T, F)  这里是 fbank: (798,128)
    """
    T, F = x.shape
    x_aug = x.clone()

    for _ in range(num_masks):
        # time mask
        t_width = random.randint(0, max_mask_t)
        t_start = random.randint(0, max(0, T - t_width))
        if t_width > 0:
            x_aug[t_start:t_start + t_width, :] = 0

        # freq mask
        f_width = random.randint(0, max_mask_f)
        f_start = random.randint(0, max(0, F - f_width))
        if f_width > 0:
            x_aug[:, f_start:f_start + f_width] = 0

    return x_aug


# ============================================================
# 2) Dataset：读取你预处理好的 (798,128) fbank.npy
# ============================================================
class ICBHINpyDataset(Dataset):
    def __init__(self, csv_path, is_train=False,
                 spec_aug=True, max_mask_t=40, max_mask_f=16, num_masks=2):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.is_train = is_train

        self.spec_aug = spec_aug
        self.max_mask_t = max_mask_t
        self.max_mask_f = max_mask_f
        self.num_masks = num_masks

        # 打印 class counts（四分类）
        labels = self.df["label"].astype(int).values
        counts = np.bincount(labels, minlength=4)
        print(f"[Dataset] {'Train' if is_train else 'Test'} | Samples: {len(self.df)} | "
              f"Counts(0/1/2/3)={counts.tolist()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        fbank = np.load(row["fbank_path"]).astype(np.float32)  # (798,128)
        y = int(row["label"])                                  # 0/1/2/3

        x = torch.from_numpy(fbank).float()                    # (798,128)

        # ✅ 训练时才做 specaug
        if self.is_train and self.spec_aug:
            x = apply_spec_augment(
                x,
                max_mask_t=self.max_mask_t,
                max_mask_f=self.max_mask_f,
                num_masks=self.num_masks
            )

        return x, torch.tensor(y, dtype=torch.long)


# ============================================================
# 3) 官方评估：Sp, Se, Sc（四分类预测，按官方异常合并算 SE）
# ============================================================
def get_icbhi_scores(preds, labels):
    # 0:Normal, 1:Crackle, 2:Wheeze, 3:Both
    hits = [0.0] * 4
    counts = [0.0] * 4

    for p, l in zip(preds, labels):
        counts[l] += 1
        if p == l:
            hits[l] += 1

    # Specificity: 正常类(0)召回率
    sp = (hits[0] / (counts[0] + 1e-10)) * 100.0

    # Sensitivity: 异常类(1,2,3)总体召回率
    se_hits = sum(hits[1:])
    se_counts = sum(counts[1:])
    se = (se_hits / (se_counts + 1e-10)) * 100.0

    # Score
    sc = (sp + se) / 2.0
    return sp, se, sc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for fbanks, labels in loader:
        fbanks = fbanks.to(device, non_blocking=True)  # (B,798,128)
        logits = model(fbanks)                         # (B,4)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())

    sp, se, sc = get_icbhi_scores(all_preds, all_labels)
    return sp, se, sc


# ============================================================
# 4) 训练主程序
# ============================================================
def main():
    # ---------- 配置 ----------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8           # ✅ 你模型比较大，建议从 8 或更小开始
    LR = 1e-4
    EPOCHS = 100

    TRAIN_CSV = "/data/dingcong/hybrid/icbhi_official_fbank/train_index.csv"
    TEST_CSV  = "/data/dingcong/hybrid/icbhi_official_fbank/test_index.csv"
    SAVE_PATH = "/data/dingcong/hybrid/best_official_score_model.pth"

    # specaug
    SPEC_AUG = True
    MAX_MASK_T = 40
    MAX_MASK_F = 16
    NUM_MASKS = 2

    # ---------- 固定随机种子 ----------
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # ---------- 数据 ----------
    train_dataset = ICBHINpyDataset(
        TRAIN_CSV, is_train=True,
        spec_aug=SPEC_AUG, max_mask_t=MAX_MASK_T, max_mask_f=MAX_MASK_F, num_masks=NUM_MASKS
    )
    test_dataset = ICBHINpyDataset(
        TEST_CSV, is_train=False,
        spec_aug=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ---------- 模型 ----------
    # ✅ 关键：num_classes=4，模型直接输出4类logits
    model = SSA_Model(
        d_model=512,
        n_layers=8,
        nhead=8,
        num_classes=4,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        local_files_only=False,
        unfreeze_projection=True,
    ).to(DEVICE)

    # ---------- loss（用 train csv 自动统计更稳） ----------
    train_counts = torch.tensor(
        train_dataset.df["label"].value_counts().sort_index().values,
        dtype=torch.float32
    )
    # 防止某类缺失导致 shape 不对
    if train_counts.numel() < 4:
        tmp = torch.zeros(4, dtype=torch.float32)
        for k, v in train_dataset.df["label"].value_counts().items():
            tmp[int(k)] = float(v)
        train_counts = tmp

    weights = (1.0 / (train_counts + 1e-6))
    weights = weights / weights.sum() * 4.0  # 均值≈1 更稳定
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    print("[INFO] train class counts :", train_counts.tolist())
    print("[INFO] train class weights:", weights.tolist())

    # ---------- optimizer/scheduler ----------
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_sc = -1.0
    best_epoch = -1

    # （可选）快速 sanity check
    xb, yb = next(iter(train_loader))
    xb = xb.to(DEVICE)
    with torch.no_grad():
        out = model(xb)
    print(f"[DEBUG] fbank batch: {xb.shape} -> logits: {out.shape} (expect Bx4)")

    # ---------- 训练循环 ----------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for fbanks, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            fbanks = fbanks.to(DEVICE, non_blocking=True)  # (B,798,128)
            labels = labels.to(DEVICE, non_blocking=True)  # (B,)

            optimizer.zero_grad(set_to_none=True)
            logits = model(fbanks)                         # (B,4)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # ---------- 测试 ----------
        sp, se, sc = evaluate(model, test_loader, DEVICE)

        print(
            f"Epoch [{epoch:03d}] "
            f"Loss: {train_loss / max(1, len(train_loader)):.4f} | "
            f"Sp: {sp:.2f}% | Se: {se:.2f}% | Score: {sc:.2f}%"
        )

        # ---------- 保存最优 ----------
        if sc > best_sc + 1e-7:
            best_sc = sc
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_sc": best_sc,
                    "best_epoch": best_epoch,
                },
                SAVE_PATH
            )
            print(f">>> ⭐ New Best Sc={best_sc:.2f}% @ epoch {best_epoch} | saved to {SAVE_PATH}")

    print(f"\n[DONE] Best official Score Sc: {best_sc:.2f}% @ epoch {best_epoch}")
    print(f"[DONE] Best checkpoint: {SAVE_PATH}")


if __name__ == "__main__":
    main()
