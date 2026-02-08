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
    # 先折算成二分类：0 -> 0, 1/2/3 -> 1
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
# 4) 构建 projection 分组 optimizer（proj 小 lr）
# ============================================================
def build_optimizer_with_proj_groups(model: nn.Module, base_lr: float, proj_lr_mult: float = 0.1,
                                     weight_decay: float = 1e-2):
    proj_params = list(model.ast_proj.proj.parameters())

    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("ast_proj.proj"):
            continue
        other_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": proj_params, "lr": base_lr * proj_lr_mult},
            {"params": other_params, "lr": base_lr},
        ],
        weight_decay=weight_decay
    )
    return optimizer

# ============================================================
# 5) 主训练程序（proj warmup + 分组lr + grad检查 + CM打印）
# ============================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE =16
    LR = 1e-4
    EPOCHS = 50

    # warmup：前 N 个 epoch 冻结 projection
    WARMUP_EPOCHS = 5
    PROJ_LR_MULT = 0.1

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

    # model
    model = SSA_Model(
        d_model=512,
        n_layers=2,
        nhead=8,
        num_classes=4,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        local_files_only=False,
        unfreeze_projection=True,
    ).to(DEVICE)

    # loss weights
    train_counts = torch.tensor(
        train_dataset.df["label"].value_counts().sort_index().values,
        dtype=torch.float32
    )
    if train_counts.numel() < 4:
        tmp = torch.zeros(4, dtype=torch.float32)
        for k, v in train_dataset.df["label"].value_counts().items():
            tmp[int(k)] = float(v)
        train_counts = tmp

    weights = (1.0 / (train_counts + 1e-6))
    weights = weights / weights.sum() * 4.0
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    print("[INFO] train class counts :", train_counts.tolist())
    print("[INFO] train class weights:", weights.tolist())
    print(f"[INFO] warmup epochs for projection: {WARMUP_EPOCHS}")
    print(f"[INFO] projection lr mult: {PROJ_LR_MULT}")

    # optimizer (group lr)
    optimizer = build_optimizer_with_proj_groups(
        model=model,
        base_lr=LR,
        proj_lr_mult=PROJ_LR_MULT,
        weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # sanity check
    xb, yb = next(iter(train_loader))
    xb = xb.to(DEVICE)
    with torch.no_grad():
        out = model(xb)
    print(f"[DEBUG] fbank batch: {xb.shape} -> logits: {out.shape} (expect Bx4)")

    best_sc = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        # ===== warmup freeze/unfreeze projection =====
        if epoch <= WARMUP_EPOCHS:
            for p in model.ast_proj.proj.parameters():
                p.requires_grad = False
            proj_status = "FROZEN"
        else:
            for p in model.ast_proj.proj.parameters():
                p.requires_grad = True
            proj_status = "TRAINABLE"

        model.train()
        train_loss = 0.0
        printed_proj_grad = False

        for fbanks, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [proj={proj_status}]"):
            fbanks = fbanks.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(fbanks)
            loss = criterion(logits, labels)
            loss.backward()

            # grad check
            if not printed_proj_grad:
                wgrad = model.ast_proj.proj.weight.grad
                ginfo = "None" if wgrad is None else f"{wgrad.abs().mean().item():.6e}"
                print(f"\n[DEBUG] epoch={epoch} proj_status={proj_status} | proj.weight.grad_mean={ginfo}\n")
                printed_proj_grad = True

            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # eval on test
        se, sp, sc, (tn, fp, fn, tp) = evaluate(model, test_loader, DEVICE)

        print(
            f"Epoch [{epoch:03d}] "
            f"Loss: {train_loss / max(1, len(train_loader)):.4f} | "
            f"SE: {se:.2f}% | SP: {sp:.2f}% | Score: {sc:.2f}% | proj={proj_status}"
        )
        print(f"Confusion Matrix (binary, Normal vs Abnormal): TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

        # save best
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
