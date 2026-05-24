#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
from pathlib import Path

# ============================================================
# 0. 项目路径
# 当前文件位置:
# /data/dingcong/hybrid/tools/test.py
#
# 项目根目录:
# /data/dingcong/hybrid
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("[DEBUG] PROJECT_ROOT:", PROJECT_ROOT)
print("[DEBUG] sys.path[0]:", sys.path[0])

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from dataset.dataloader import build_dataloader
from mymodels.model import TimeFrequencyEncoder


# ============================================================
# 1. Encoder + Logistic Regression Classifier
# ============================================================
class TimeFrequencyLogisticModel(nn.Module):
    """
    输入:
        tokens: [B, 948, 768]

    输出:
        logits: [B, 4]
    """

    def __init__(
        self,
        num_classes=4,
        token_dim=768,
        freq_patches=12,
        time_patches=79,
        time_depth=2,
        freq_depth=2,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = TimeFrequencyEncoder(
            token_dim=token_dim,
            freq_patches=freq_patches,
            time_patches=time_patches,
            time_depth=time_depth,
            freq_depth=freq_depth,
            num_heads=num_heads,
            dropout=dropout
        )

        # 逻辑回归分类器
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, x):
        feature = self.encoder(x)          # [B, 768]
        logits = self.classifier(feature)  # [B, 4]
        return logits


# ============================================================
# 2. 固定随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. ICBHI Score: SE / SP / Score
#    normal = class 0
#    abnormal = class 1, 2, 3
# ============================================================
def icbhi_from_probs(p_abn, labels_4, thr):
    """
    p_abn:
        abnormal probability = 1 - P(class 0)

    labels_4:
        0: normal
        1: crackle
        2: wheeze
        3: both
    """

    labels_bin = np.array(
        [0 if int(l) == 0 else 1 for l in labels_4],
        dtype=np.int64
    )

    preds_bin = (np.array(p_abn) >= thr).astype(np.int64)

    cm = confusion_matrix(labels_bin, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = 100.0 * tn / (tn + fp + 1e-10)
    se = 100.0 * tp / (tp + fn + 1e-10)
    score = 0.5 * (se + sp)

    return se, sp, score, (tn, fp, fn, tp)


@torch.no_grad()
def evaluate(model, loader, device, thr_grid=None):
    model.eval()

    all_labels = []
    all_preds = []
    all_p_abn = []

    for tokens, labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(tokens)
        prob = torch.softmax(logits, dim=1)

        preds = torch.argmax(prob, dim=1)

        p0 = prob[:, 0].detach().cpu().numpy()
        p_abn = 1.0 - p0

        all_p_abn.append(p_abn)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    all_p_abn = np.concatenate(all_p_abn, axis=0)

    if thr_grid is None:
        thr_grid = np.linspace(0.0, 1.0, 201)

    best_thr = 0.5
    best_score = -1.0
    best_se = 0.0
    best_sp = 0.0
    best_cm = (0, 0, 0, 0)

    for thr in thr_grid:
        se, sp, score, cm = icbhi_from_probs(
            all_p_abn,
            all_labels,
            float(thr)
        )

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_se = se
            best_sp = sp
            best_cm = cm

    acc = accuracy_score(all_labels, all_preds)

    try:
        macro_f1 = f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )
    except Exception:
        macro_f1 = 0.0

    pred_abn_ratio = 100.0 * float((all_p_abn >= best_thr).mean())

    return {
        "thr": best_thr,
        "se": best_se,
        "sp": best_sp,
        "score": best_score,
        "cm": best_cm,
        "acc": acc,
        "macro_f1": macro_f1,
        "pred_abn_ratio": pred_abn_ratio
    }


# ============================================================
# 4. Cosine LR
# ============================================================
def cosine_lr(epoch, total_epochs, base_lr):
    if total_epochs <= 1:
        return base_lr

    t = (epoch - 1) / (total_epochs - 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


# ============================================================
# 5. 主训练程序
# ============================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_CSV = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/train_index.csv"
    TEST_CSV = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/test_index.csv"

    SAVE_PATH = "/data/dingcong/hybrid/best_time_frequency_logistic_model.pth"

    # 先用 1 跑通，确认没问题后改成 50
    EPOCHS = 1

    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8

    LR = 5e-5
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4

    SEED = 42
    set_seed(SEED)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("=" * 80)
    print("[INFO] Training Time-Frequency Encoder + Logistic Regression")
    print("[INFO] Device:", DEVICE)
    print("[INFO] Train CSV:", TRAIN_CSV)
    print("[INFO] Test CSV :", TEST_CSV)
    print("[INFO] Save Path:", SAVE_PATH)
    print("[INFO] Epochs:", EPOCHS)
    print("[INFO] Batch Size:", BATCH_SIZE)
    print("[INFO] Accumulation Steps:", ACCUMULATION_STEPS)
    print("[INFO] Effective Batch Size:", BATCH_SIZE * ACCUMULATION_STEPS)
    print("=" * 80)

    # =====================================================
    # DataLoader
    # =====================================================
    train_loader = build_dataloader(
        csv_path=TRAIN_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    test_loader = build_dataloader(
        csv_path=TEST_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    print("[INFO] Train batches:", len(train_loader))
    print("[INFO] Test batches :", len(test_loader))

    # =====================================================
    # Model
    # =====================================================
    model = TimeFrequencyLogisticModel(
        num_classes=4,
        token_dim=768,
        freq_patches=12,
        time_patches=79,
        time_depth=2,
        freq_depth=2,
        num_heads=8,
        dropout=0.1
    ).to(DEVICE)

    # =====================================================
    # Loss weights
    # =====================================================
    train_df = pd.read_csv(TRAIN_CSV)

    train_counts = torch.zeros(4, dtype=torch.float32)

    for k, v in train_df["label"].value_counts().items():
        train_counts[int(k)] = float(v)

    weights = 1.0 / (train_counts + 1e-6)
    weights = weights / weights.sum() * 4.0

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    print("[INFO] train class counts :", train_counts.tolist())
    print("[INFO] train class weights:", weights.tolist())

    # =====================================================
    # Optimizer + AMP
    # =====================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # =====================================================
    # Sanity check
    # =====================================================
    tokens, labels = next(iter(train_loader))
    tokens = tokens.to(DEVICE)

    with torch.no_grad():
        logits = model(tokens)

    print("[DEBUG] tokens:", tokens.shape)
    print("[DEBUG] logits:", logits.shape)

    if logits.shape[-1] != 4:
        raise ValueError(f"logits shape 错误: {logits.shape}, 期望 [B, 4]")

    # =====================================================
    # Training
    # =====================================================
    best_score = -1.0
    best_epoch = -1
    best_state_dict = None
    best_result = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        lr_now = cosine_lr(epoch, EPOCHS, LR)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            ncols=120
        )

        for step, (tokens, labels) in enumerate(pbar):
            tokens = tokens.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(tokens)
                loss = criterion(logits, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_value = loss.item() * ACCUMULATION_STEPS
            train_loss += loss_value

            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr": f"{lr_now:.2e}"
            })

        # 如果最后一个 batch 没有刚好满足 accumulation，也更新一次
        if (step + 1) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = train_loss / max(1, len(train_loader))

        # =====================================================
        # Evaluation
        # =====================================================
        result = evaluate(
            model=model,
            loader=test_loader,
            device=DEVICE
        )

        tn, fp, fn, tp = result["cm"]

        print(
            f"\nEpoch [{epoch:03d}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"Score: {result['score']:.2f} | "
            f"SE: {result['se']:.2f} | "
            f"SP: {result['sp']:.2f} | "
            f"thr={result['thr']:.3f} | "
            f"Acc: {result['acc']:.4f} | "
            f"Macro-F1: {result['macro_f1']:.4f} | "
            f"PredAbn: {result['pred_abn_ratio']:.2f}% | "
            f"lr={lr_now:.2e}"
        )

        print(
            f"Confusion Matrix binary: "
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
        )

        # =====================================================
        # Save best in memory
        # =====================================================
        if result["score"] > best_score:
            best_score = result["score"]
            best_epoch = epoch

            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            best_result = result

            print(
                f">>> New Best Score={best_score:.2f} "
                f"@ Epoch {best_epoch}"
            )

    # =====================================================
    # Save final best checkpoint
    # =====================================================
    if best_state_dict is not None:
        tn, fp, fn, tp = best_result["cm"]

        torch.save(
            {
                "model": best_state_dict,
                "best_score": float(best_score),
                "best_epoch": int(best_epoch),
                "best_thr": float(best_result["thr"]),
                "best_se": float(best_result["se"]),
                "best_sp": float(best_result["sp"]),
                "best_acc": float(best_result["acc"]),
                "best_macro_f1": float(best_result["macro_f1"]),
                "best_pred_abn_ratio": float(best_result["pred_abn_ratio"]),
                "best_cm": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                },
                "config": {
                    "epochs": int(EPOCHS),
                    "batch_size": int(BATCH_SIZE),
                    "accumulation_steps": int(ACCUMULATION_STEPS),
                    "effective_batch_size": int(BATCH_SIZE * ACCUMULATION_STEPS),
                    "lr": float(LR),
                    "weight_decay": float(WEIGHT_DECAY),
                    "token_dim": 768,
                    "freq_patches": 12,
                    "time_patches": 79,
                    "time_depth": 2,
                    "freq_depth": 2,
                    "num_heads": 8,
                    "dropout": 0.1,
                    "seed": int(SEED)
                }
            },
            SAVE_PATH
        )

        print("=" * 80)
        print("[DONE] Saved best checkpoint:", SAVE_PATH)
        print(
            f"[DONE] Best @ Epoch {best_epoch}: "
            f"Score={best_score:.2f} | "
            f"SE={best_result['se']:.2f} | "
            f"SP={best_result['sp']:.2f} | "
            f"thr={best_result['thr']:.3f} | "
            f"Acc={best_result['acc']:.4f} | "
            f"Macro-F1={best_result['macro_f1']:.4f}"
        )
        print("=" * 80)

    else:
        print("[WARN] No best model was saved.")


if __name__ == "__main__":
    main()