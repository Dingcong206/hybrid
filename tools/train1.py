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
from sklearn.metrics import confusion_matrix

# ============================================================
# 0) 项目路径与导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import SSA_Model


# ============================================================
# 1) SpecAugment（对 fbank: T x F 遮挡）
# ============================================================
def apply_spec_augment(x, max_mask_t=40, max_mask_f=16, num_masks=2):
    T, F = x.shape
    x_aug = x.clone()

    for _ in range(num_masks):
        t_width = random.randint(0, max_mask_t)
        t_start = random.randint(0, max(0, T - t_width))
        if t_width > 0:
            x_aug[t_start:t_start + t_width, :] = 0

        f_width = random.randint(0, max_mask_f)
        f_start = random.randint(0, max(0, F - f_width))
        if f_width > 0:
            x_aug[:, f_start:f_start + f_width] = 0

    return x_aug


# ============================================================
# 2) Dataset：读取 (798,128) fbank.npy
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
        x = torch.from_numpy(fbank).float()

        if self.is_train and self.spec_aug:
            x = apply_spec_augment(
                x,
                max_mask_t=self.max_mask_t,
                max_mask_f=self.max_mask_f,
                num_masks=self.num_masks
            )
        return x, torch.tensor(y, dtype=torch.long)


# ============================================================
# 3) ICBHI 官方指标 + 混淆矩阵（折算后二分类）
# ============================================================
def icbhi_from_preds(preds, labels):
    """
    preds, labels: list of 0/1/2/3
    返回: SE, SP, Score, (TN, FP, FN, TP)
    """
    preds_bin = [0 if p == 0 else 1 for p in preds]
    labels_bin = [0 if l == 0 else 1 for l in labels]

    cm = confusion_matrix(labels_bin, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = 100.0 * tn / (tn + fp + 1e-10)
    se = 100.0 * tp / (tp + fn + 1e-10)
    score = (sp + se) / 2.0

    return se, sp, score, (tn, fp, fn, tp)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for fbanks, labels in loader:
        fbanks = fbanks.to(device, non_blocking=True)
        logits = model(fbanks)  # (B,4)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())

    se, sp, score, cm = icbhi_from_preds(all_preds, all_labels)
    return se, sp, score, cm


# ============================================================
# 4) 冻结 AST（包括 projection）
# ============================================================
def freeze_ast_all(model: nn.Module):
    if hasattr(model, "ast_proj") and hasattr(model.ast_proj, "proj"):
        for p in model.ast_proj.proj.parameters():
            p.requires_grad = False


# ============================================================
# 5) 主训练程序（冻结AST + 输出SE/SP/Score + 只保存最终最优一次）
# ============================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 50

    TRAIN_CSV = "/data/dingcong/hybrid/icbhi_official_fbank/train_index.csv"
    TEST_CSV  = "/data/dingcong/hybrid/icbhi_official_fbank/test_index.csv"
    SAVE_PATH = "/data/dingcong/hybrid/best_official_score_model.pth"

    SPEC_AUG = True
    MAX_MASK_T = 40
    MAX_MASK_F = 16
    NUM_MASKS = 2

    # seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # data
    train_dataset = ICBHINpyDataset(
        TRAIN_CSV, is_train=True,
        spec_aug=SPEC_AUG, max_mask_t=MAX_MASK_T, max_mask_f=MAX_MASK_F, num_masks=NUM_MASKS
    )
    test_dataset = ICBHINpyDataset(TEST_CSV, is_train=False, spec_aug=False)

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

    # model（冻结 AST 前端）
    model = SSA_Model(
        d_model=512,
        n_layers=2,
        nhead=8,
        num_classes=4,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        local_files_only=False,
        unfreeze_projection=False,
    ).to(DEVICE)
    freeze_ast_all(model)
    print("[INFO] AST(projection) is FROZEN: True")

    # loss weights
    vc = train_dataset.df["label"].value_counts()
    train_counts = torch.zeros(4, dtype=torch.float32)
    for k, v in vc.items():
        train_counts[int(k)] = float(v)

    weights = (1.0 / (train_counts + 1e-6))
    weights = weights / weights.sum() * 4.0
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    print("[INFO] train class counts :", train_counts.tolist())
    print("[INFO] train class weights:", weights.tolist())

    # optimizer：只优化 requires_grad=True 的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # sanity check
    xb, _ = next(iter(train_loader))
    xb = xb.to(DEVICE)
    with torch.no_grad():
        out = model(xb)
    print(f"[DEBUG] fbank batch: {xb.shape} -> logits: {out.shape} (expect Bx4)")

    best_sc = -1.0
    best_epoch = -1
    best_metrics = None
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for fbanks, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [AST=FROZEN]"):
            fbanks = fbanks.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(fbanks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # eval on test
        se, sp, sc, (tn, fp, fn, tp) = evaluate(model, test_loader, DEVICE)

        # 每轮都输出 Score + SE + SP（你要求的）
        print(
            f"Epoch [{epoch:03d}] "
            f"Loss: {train_loss / max(1, len(train_loader)):.4f} | "
            f"Score: {sc:.1f} | SE: {se:.1f} | SP: {sp:.1f} | AST=FROZEN"
        )
        print(f"Confusion Matrix (binary, Normal vs Abnormal): TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

        # 只更新“内存中的最佳”，不落盘
        if sc > best_sc + 1e-7:
            best_sc = sc
            best_epoch = epoch
            best_metrics = (se, sp, sc, tn, fp, fn, tp)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f">>> ⭐ New Best (in-memory) Score={best_sc:.1f} @ epoch {best_epoch} (NOT saved yet)")

    # 训练结束后，只保存一次：最终最优
    if best_state_dict is not None:
        se, sp, sc, tn, fp, fn, tp = best_metrics
        torch.save(
            {
                "model": best_state_dict,
                "best_sc": float(best_sc),
                "best_epoch": int(best_epoch),
                "best_se": float(se),
                "best_sp": float(sp),
                "best_cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
            SAVE_PATH
        )
        print(f"\n[DONE] Saved ONLY ONCE: best checkpoint -> {SAVE_PATH}")
        print(f"[DONE] Best @ epoch {best_epoch}: Score={sc:.1f} | SE={se:.1f} | SP={sp:.1f} | TN={tn} FP={fp} FN={fn} TP={tp}")
    else:
        print("\n[WARN] best_state_dict is None (unexpected). No checkpoint saved.")


if __name__ == "__main__":
    main()
