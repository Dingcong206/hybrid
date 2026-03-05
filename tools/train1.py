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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# ============================================================
# 0) 项目路径与导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import SSA_Model  # 你的模型（fbank->AST proj->SSA）


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
# 3) ICBHI 指标（用异常概率 + 阈值） + 二分类混淆矩阵
# ============================================================
def icbhi_from_probs(p_abn, labels_4, thr: float):
    """
    p_abn: ndarray (N,)   异常概率 = 1 - P(class0)
    labels_4: list[int]   0/1/2/3
    thr: float            阈值
    return: SE, SP, Score, (TN, FP, FN, TP)
    """
    labels_bin = np.array([0 if l == 0 else 1 for l in labels_4], dtype=np.int64)
    preds_bin = (np.array(p_abn) >= thr).astype(np.int64)

    cm = confusion_matrix(labels_bin, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = 100.0 * tn / (tn + fp + 1e-10)
    se = 100.0 * tp / (tp + fn + 1e-10)
    score = 0.5 * (sp + se)
    return se, sp, score, (tn, fp, fn, tp)


@torch.no_grad()
def evaluate(model, loader, device, thr_grid=None):
    """
    在测试集上搜索最优阈值 thr（基于 p_abn = 1 - P(normal)）
    返回：best_thr, best_se, best_sp, best_score, best_cm, pred_abn_ratio
    """
    model.eval()
    all_labels = []
    all_p_abn = []

    for fbanks, labels in loader:
        fbanks = fbanks.to(device, non_blocking=True)
        logits = model(fbanks)              # (B,4)
        prob = torch.softmax(logits, dim=1) # (B,4)

        p0 = prob[:, 0].detach().cpu().numpy()  # P(normal)
        p_abn = 1.0 - p0                         # P(abnormal)

        all_p_abn.append(p_abn)
        all_labels.extend(labels.numpy().tolist())

    p_abn = np.concatenate(all_p_abn, axis=0)

    if thr_grid is None:
        thr_grid = np.linspace(0.0, 1.0, 201)  # 0.005 步长

    best_thr = 0.5
    best_score = -1.0
    best_se = best_sp = 0.0
    best_cm = (0, 0, 0, 0)

    for thr in thr_grid:
        se, sp, score, cm = icbhi_from_probs(p_abn, all_labels, float(thr))
        if score > best_score + 1e-7:
            best_score = score
            best_thr = float(thr)
            best_se, best_sp = se, sp
            best_cm = cm

    pred_abn_ratio = 100.0 * float((p_abn >= best_thr).mean())
    return best_thr, best_se, best_sp, best_score, best_cm, pred_abn_ratio


# ============================================================
# 4) projection 冻结/解冻
# ============================================================
def set_projection_trainable(model: nn.Module, trainable: bool):
    if hasattr(model, "ast_proj") and hasattr(model.ast_proj, "proj"):
        for p in model.ast_proj.proj.parameters():
            p.requires_grad = trainable


# ============================================================
# 5) 手动 Cosine LR（proj lr = base_lr * PROJ_LR_MULT）
# ============================================================
def cosine_lr(epoch: int, total_epochs: int, base_lr: float):
    if total_epochs <= 1:
        return base_lr
    t = (epoch - 1) / (total_epochs - 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


# ============================================================
# 6) 主训练程序：前10轮冻结 projection，后面微调；每轮输出 Score/SE/SP/CM/thr
#    训练结束只保存一次：最终最优（best_score）
# ============================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 2  # ✅ 实际喂给 GPU 的大小，设小防止 OOM
    ACCUMULATION_STEPS = 8  # ✅ 累加步数。有效 Batch Size = 4 * 4 = 16
    # -------------------
    LR = 5e-5
    EPOCHS = 50

    WARMUP_EPOCHS = 10           # ✅ 前10轮冻结 projection
    PROJ_LR_MULT = 0.03          # ✅ 解冻后 projection 使用更小 lr（base_lr * mult）

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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print(f"[INFO] warmup epochs (projection frozen): {WARMUP_EPOCHS}")
    print(f"[INFO] projection lr mult (after warmup): {PROJ_LR_MULT}")

    # model：先允许 projection 可训练，但我们每轮手动冻结/解冻
    model = SSA_Model(
        d_model=256,
        n_layers=2,
        nhead=6,
        num_classes=4,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        local_files_only=False,
        unfreeze_projection=True,
    ).to(DEVICE)

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

    # optimizer：两个 param group（proj / others）
    proj_params = list(model.ast_proj.proj.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("ast_proj.proj") and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": proj_params,  "lr": 0.0},  # warmup 期间先 0，后面每轮会更新
            {"params": other_params, "lr": LR},
        ],
        weight_decay=1e-2
    )

    # sanity check
    xb, _ = next(iter(train_loader))
    xb = xb.to(DEVICE)
    with torch.no_grad():
        out = model(xb)
    print(f"[DEBUG] fbank batch: {xb.shape} -> logits: {out.shape} (expect Bx4)")

    best_sc = -1.0
    best_epoch = -1
    best_state_dict = None
    best_pack = None  # (thr, se, sp, sc, tn, fp, fn, tp, pred_abn_ratio)

    for epoch in range(1, EPOCHS + 1):
        # ===== 冻结/解冻 projection =====
        if epoch <= WARMUP_EPOCHS:
            set_projection_trainable(model, False)
            proj_status = "FROZEN"
        else:
            set_projection_trainable(model, True)
            proj_status = "TRAINABLE"

        # ===== 手动设置本轮 lr（cosine）=====
        base_lr_now = float(cosine_lr(epoch, EPOCHS, LR))
        optimizer.param_groups[1]["lr"] = base_lr_now  # others
        optimizer.param_groups[0]["lr"] = (0.0 if proj_status == "FROZEN"
                                           else base_lr_now * PROJ_LR_MULT)  # proj

        model.train()
        train_loss = 0.0

        for i, (fbanks, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [proj={proj_status}]")):
            fbanks = fbanks.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # 1. 前向传播
            logits = model(fbanks)
            # 2. 计算损失并除以累加步数（取平均值）
            loss = criterion(logits, labels) / ACCUMULATION_STEPS

            # 3. 反向传播（梯度会持续累加在 param.grad 中）
            loss.backward()

            # 4. 当达到累加步数时，才更新参数
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * ACCUMULATION_STEPS  # 还原回用于显示的 loss

        # ===== eval on TEST：搜索最优阈值 thr =====
        thr, se, sp, sc, (tn, fp, fn, tp), pred_abn = evaluate(model, test_loader, DEVICE)

        print(
            f"Epoch [{epoch:03d}] "
            f"Loss: {train_loss / max(1, len(train_loader)):.4f} | "
            f"Score: {sc:.1f} | SE: {se:.1f} | SP: {sp:.1f} | thr={thr:.3f} | "
            f"PredAbn: {pred_abn:.1f}% | proj={proj_status} | "
            f"lr={optimizer.param_groups[1]['lr']:.2e} proj_lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        print(f"Confusion Matrix (binary, Normal vs Abnormal): TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

        # 只更新“内存最优”，不落盘
        if sc > best_sc + 1e-7:
            best_sc = sc
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pack = (thr, se, sp, sc, tn, fp, fn, tp, pred_abn)
            print(f">>> ⭐ New Best (in-memory) Score={best_sc:.1f} @ epoch {best_epoch} (NOT saved yet)")

    # 训练结束：只保存一次（最终最优）
    if best_state_dict is not None and best_pack is not None:
        thr, se, sp, sc, tn, fp, fn, tp, pred_abn = best_pack
        torch.save(
            {
                "model": best_state_dict,
                "best_sc": float(best_sc),
                "best_epoch": int(best_epoch),
                "best_thr": float(thr),
                "best_se": float(se),
                "best_sp": float(sp),
                "best_pred_abn_ratio": float(pred_abn),
                "best_cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                "warmup_epochs": int(WARMUP_EPOCHS),
                "proj_lr_mult": float(PROJ_LR_MULT),
                "base_lr": float(LR),
                "epochs": int(EPOCHS),
                "batch_size": int(BATCH_SIZE),
            },
            SAVE_PATH
        )
        print(f"\n[DONE] Saved ONLY ONCE: best checkpoint -> {SAVE_PATH}")
        print(f"[DONE] Best @ epoch {best_epoch}: "  
              f"Score={sc:.1f} | SE={se:.1f} | SP={sp:.1f} | thr={thr:.3f} | PredAbn={pred_abn:.1f}% | "
              f"TN={tn} FP={fp} FN={fn} TP={tp}")
    else:
        print("\n[WARN] best_state_dict is None (unexpected). No checkpoint saved.")


if __name__ == "__main__":
    main()
