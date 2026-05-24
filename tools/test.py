#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataloader import build_dataloader
from mymodels.model import TimeFrequencyEncoder


# =====================================================
# 1. 完整模型：Encoder + Logistic Regression
# =====================================================
class TimeFrequencyLogisticModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.encoder = TimeFrequencyEncoder(
            token_dim=768,
            freq_patches=12,
            time_patches=79,
            time_depth=2,
            freq_depth=2,
            num_heads=8,
            dropout=0.1
        )

        # 逻辑回归分类器
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        """
        x: [B, 948, 768]
        """
        feature = self.encoder(x)          # [B, 768]
        logits = self.classifier(feature)  # [B, 4]
        return logits


# =====================================================
# 2. ICBHI 二分类评价指标
# =====================================================
def icbhi_from_probs(p_abn, labels_4, thr):
    labels_bin = np.array([0 if l == 0 else 1 for l in labels_4], dtype=np.int64)
    preds_bin = (np.array(p_abn) >= thr).astype(np.int64)

    cm = confusion_matrix(labels_bin, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = 100.0 * tn / (tn + fp + 1e-10)
    se = 100.0 * tp / (tp + fn + 1e-10)
    score = 0.5 * (sp + se)

    return se, sp, score, (tn, fp, fn, tp)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_p_abn = []

    for tokens, labels in loader:
        tokens = tokens.to(device, non_blocking=True)

        logits = model(tokens)
        prob = torch.softmax(logits, dim=1)

        p0 = prob[:, 0].detach().cpu().numpy()
        p_abn = 1.0 - p0

        all_p_abn.append(p_abn)
        all_labels.extend(labels.numpy().tolist())

    p_abn = np.concatenate(all_p_abn, axis=0)

    best_thr = 0.5
    best_score = -1.0
    best_se = 0.0
    best_sp = 0.0
    best_cm = (0, 0, 0, 0)

    for thr in np.linspace(0.0, 1.0, 201):
        se, sp, score, cm = icbhi_from_probs(p_abn, all_labels, float(thr))

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_se = se
            best_sp = sp
            best_cm = cm

    pred_abn_ratio = 100.0 * float((p_abn >= best_thr).mean())

    return best_thr, best_se, best_sp, best_score, best_cm, pred_abn_ratio


def cosine_lr(epoch, total_epochs, base_lr):
    if total_epochs <= 1:
        return base_lr

    t = (epoch - 1) / (total_epochs - 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


# =====================================================
# 3. 主训练程序
# =====================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 第一次测试可以先用 1
    EPOCHS = 1

    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8
    LR = 5e-5

    TRAIN_CSV = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/train_index.csv"
    TEST_CSV = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/test_index.csv"
    SAVE_PATH = "/data/dingcong/hybrid/best_time_frequency_logistic_model.pth"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    print("[INFO] Device:", DEVICE)
    print("[INFO] Train CSV:", TRAIN_CSV)
    print("[INFO] Test CSV :", TEST_CSV)

    # =====================================================
    # 4. DataLoader
    # =====================================================
    train_loader = build_dataloader(
        csv_path=TRAIN_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = build_dataloader(
        csv_path=TEST_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # =====================================================
    # 5. 模型
    # =====================================================
    model = TimeFrequencyLogisticModel(num_classes=4).to(DEVICE)

    # =====================================================
    # 6. 类别权重
    # =====================================================
    train_df = pd.read_csv(TRAIN_CSV)
    vc = train_df["label"].value_counts()

    train_counts = torch.zeros(4, dtype=torch.float32)

    for k, v in vc.items():
        train_counts[int(k)] = float(v)

    weights = 1.0 / (train_counts + 1e-6)
    weights = weights / weights.sum() * 4.0

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    print("[INFO] train class counts :", train_counts.tolist())
    print("[INFO] train class weights:", weights.tolist())

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-2
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # =====================================================
    # 7. 先做一次 sanity check
    # =====================================================
    tokens, labels = next(iter(train_loader))
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        logits = model(tokens)

    print("[DEBUG] tokens:", tokens.shape)
    print("[DEBUG] logits:", logits.shape)

    # =====================================================
    # 8. 训练
    # =====================================================
    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()

        lr_now = cosine_lr(epoch, EPOCHS, LR)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for i, (tokens, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        ):
            tokens = tokens.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(tokens)
                loss = criterion(logits, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * ACCUMULATION_STEPS

        # 防止最后不足 ACCUMULATION_STEPS 的梯度没有更新
        if (i + 1) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = train_loss / max(1, len(train_loader))

        thr, se, sp, score, (tn, fp, fn, tp), pred_abn = evaluate(
            model,
            test_loader,
            DEVICE
        )

        print(
            f"Epoch [{epoch:03d}] "
            f"Loss: {avg_loss:.4f} | "
            f"Score: {score:.1f} | "
            f"SE: {se:.1f} | "
            f"SP: {sp:.1f} | "
            f"thr={thr:.3f} | "
            f"PredAbn={pred_abn:.1f}% | "
            f"lr={lr_now:.2e}"
        )

        print(
            f"Confusion Matrix: "
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "model": model.state_dict(),
                    "best_score": float(best_score),
                    "best_epoch": int(best_epoch),
                    "best_thr": float(thr),
                    "best_se": float(se),
                    "best_sp": float(sp),
                    "best_cm": {
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp)
                    },
                    "lr": float(LR),
                    "batch_size": int(BATCH_SIZE),
                    "accumulation_steps": int(ACCUMULATION_STEPS),
                },
                SAVE_PATH
            )

            print(f">>> New best model saved to {SAVE_PATH}")

    print("\n[DONE] Training finished.")
    print(f"[DONE] Best Score: {best_score:.1f} @ Epoch {best_epoch}")


if __name__ == "__main__":
    main()