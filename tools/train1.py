#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# ============================================================
# 0) 路径与模型导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from mymodels import build_model


# ============================================================
# 1) SpecAugment（针对 T x D 的 T 和 D 维度遮挡）
# ============================================================
def apply_spec_augment(x, max_mask_t=10, max_mask_f=4, num_masks=2):
    """
    x: torch.Tensor (T, D)
    """
    T, D = x.shape
    x_aug = x.clone()
    for _ in range(num_masks):
        # time mask
        t_width = random.randint(0, max_mask_t)
        t_start = random.randint(0, max(0, T - t_width))
        if t_width > 0:
            x_aug[t_start:t_start + t_width, :] = 0

        # freq mask
        f_width = random.randint(0, max_mask_f)
        f_start = random.randint(0, max(0, D - f_width))
        if f_width > 0:
            x_aug[:, f_start:f_start + f_width] = 0
    return x_aug


# ============================================================
# 2) Dataset：二分类逻辑 (0=Normal, 1=Abnormal)
# ============================================================
class TokenNPYBinaryDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = False, args=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.is_train = is_train
        self.args = args

        # 原四分类 label (0/1/2/3)
        self.y4 = self.df["label"].astype(int).values
        # 二分类转换：Label > 0 统统视为 1 (Abnormal)
        self.y = (self.y4 > 0).astype(np.int64)

        self.class_counts = np.bincount(self.y, minlength=2)
        print(f"[Dataset] {'Train' if is_train else 'Test'} | Samples: {len(self.df)} | "
              f"Counts(Normal/Abnormal): {self.class_counts.tolist()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = torch.from_numpy(np.load(row["tokens_path"])).float()  # (T, D)
        y = int(self.y[idx])

        return x, torch.tensor(y, dtype=torch.long)


def collate_pad(batch):
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    T_max = max(lens)
    B = len(xs)

    x_pad = torch.zeros(B, T_max, D, dtype=torch.float32)
    mask = torch.ones(B, T_max, dtype=torch.bool)  # True=padding, False=valid

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        mask[i, :T] = False

    return x_pad, mask, torch.stack(ys).view(-1)


# ============================================================
# 3) 评价指标：ICBHI Score（固定决策：argmax，不扫阈值）
# ============================================================
def icbhi_score_from_cm(tn, fp, fn, tp):
    eps = 1e-10
    sp = 100.0 * (tn / (tn + fp + eps))
    se = 100.0 * (tp / (tp + fn + eps))
    return sp, se, (sp + se) / 2.0


@torch.no_grad()
def evaluate_binary_argmax(backbone, classifier, loader, device):
    """
    与作者思路对齐：
    - 不扫阈值
    - 二分类用 2-logit softmax + argmax 固定决策
    """
    backbone.eval()
    classifier.eval()

    preds, trues = [], []
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        feat = backbone(x, mask=mask)
        logits = classifier(feat)              # (B,2)
        pred = torch.argmax(logits, dim=1)     # (B,)

        preds.append(pred.cpu())
        trues.append(y.cpu())

    pred = torch.cat(preds).numpy().astype(np.int64)
    t = torch.cat(trues).numpy().astype(np.int64)

    cm = confusion_matrix(t, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp, se, score = icbhi_score_from_cm(tn, fp, fn, tp)
    return {"ICBHI": score, "SP": sp, "SE": se, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ============================================================
# 4) 主训练程序（对齐作者范式 + early stop）
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # ============ 路径 & 基础设置 ============
    parser.add_argument("--root", type=str,
                        default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens")
    parser.add_argument("--save_dir", type=str,
                        default="/data/dingcong/hybrid/checkpoints_icbhi_4cls_author_style")

    # ============ 训练超参数（已帮你调好默认） ============
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)   # 10轮不提升早停

    # ============ AMP ============
    parser.add_argument("--amp", action="store_true", default=True)

    # ============ 任务形式（关键：已按你要求默认开启） ============
    parser.add_argument(
        "--two_cls_eval",
        action="store_true",
        default=True,   # ✅ 默认：官方折算二分类 SP/SE/ICBHI
        help="True=官方 normal vs abnormal 折算；False=严格四分类命中"
    )

    # ============ 类别不平衡（可选） ============
    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        default=Ture,  # 你可以以后手动开
        help="是否使用类别加权 CrossEntropy"
    )

    # ============ 模型结构（你的 backbone） ============
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=2)


    args = parser.parse_args()

    # seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ===== Data =====
    train_ds = TokenNPYBinaryDataset(os.path.join(args.root, "train_index.csv"), is_train=True, args=args)
    test_ds  = TokenNPYBinaryDataset(os.path.join(args.root, "test_index.csv"),  is_train=False, args=args)

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_pad, drop_last=True)
    dl_test  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_pad)

    # ===== Model =====
    backbone = build_model(
        in_dim=args.in_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        nhead=args.nhead
    ).to(device)

    # 方案1：2类输出 + CrossEntropy（对齐作者 argmax）
    classifier = nn.Linear(backbone.final_feat_dim, 2).to(device)

    # ===== Loss =====
    if args.weighted_loss:
        n0, n1 = train_ds.class_counts
        # 频次反比，类似作者 weighted_loss
        w0 = (n0 + n1) / (2.0 * max(n0, 1))
        w1 = (n0 + n1) / (2.0 * max(n1, 1))
        class_w = torch.tensor([w0, w1], device=device, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
        print(f"[INFO] weighted_loss ON | class_weight = {class_w.detach().cpu().tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=1e-2
    )

    # scheduler：你原来是 cosine on steps，这里保留同风格
    total_steps = max(1, args.epochs * (len(dl_train) // max(1, args.accum_steps)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # ===== Early stop tracking =====
    best_score = -1.0
    best_epoch = -1
    bad_count = 0

    print("\n🚀 Start Training (Author-aligned): Train -> Evaluate on Official Test each epoch (no thr-scan)\n")

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()

        optimizer.zero_grad()
        train_loss_sum = 0.0
        step_count = 0

        for i, (x, mask, y) in enumerate(dl_train):
            x, mask, y = x.to(device), mask.to(device), y.to(device)  # y long

            with torch.autocast(device_type="cuda", enabled=args.amp):
                feat = backbone(x, mask=mask)
                logits = classifier(feat)                 # (B,2)
                loss = loss_fn(logits, y) / args.accum_steps

            scaler.scale(loss).backward()
            train_loss_sum += loss.item() * args.accum_steps

            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

        # ===== Evaluate on official test (author style) =====
        res = evaluate_binary_argmax(backbone, classifier, dl_test, device)
        score = res["ICBHI"]

        # 对齐作者：score提升且se>5才认为有效（防止全判normal等极端情况）
        improved = (score > best_score + 1e-7) and (res["SE"] > 5.0)

        if improved:
            best_score = score
            best_epoch = epoch
            bad_count = 0

            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "classifier": classifier.state_dict(),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                os.path.join(args.save_dir, "best_model.pt")
            )
            status = "⭐"
        else:
            bad_count += 1
            status = " "

        avg_train_loss = train_loss_sum / max(1, len(dl_train))
        print(
            f"{status} Epoch {epoch:03d} | Loss: {avg_train_loss:.4f} | "
            f"Test ICBHI: {score:.4f} (SE: {res['SE']:.2f}, SP: {res['SP']:.2f}) | "
            f"TP {res['TP']} TN {res['TN']} FP {res['FP']} FN {res['FN']} | "
            f"bad_count={bad_count}/{args.patience}"
        )

        if bad_count >= args.patience:
            print(f"\n⛔ Early stop at epoch {epoch}. Best Score: {best_score:.4f} @ epoch {best_epoch}")
            break

    print(f"\n✅ Done. Best Score: {best_score:.4f} @ epoch {best_epoch}")
    print(f"📌 Best checkpoint saved to: {os.path.join(args.save_dir, 'best_model.pt')}\n")

    # =========================
    # Final evaluation (load best_model.pt -> eval on test once)
    # =========================
    # =========================
    # Final evaluation (load best_model.pt -> eval on test once)
    # =========================
    best_path = os.path.join(args.save_dir, "best_model.pt")
    if os.path.isfile(best_path):
        print("\n🧪 Final Evaluation on Test (using best_model.pt)\n")

        # 🔥 关键修复：显式关闭 weights_only
        ckpt = torch.load(best_path, map_location=device, weights_only=False)

        backbone.load_state_dict(ckpt["backbone"], strict=True)
        classifier.load_state_dict(ckpt["classifier"], strict=True)

        final_res = evaluate_binary_argmax(backbone, classifier, dl_test, device)

        print(
            f"🔥 FINAL TEST RESULT | "
            f"ICBHI: {final_res['ICBHI']:.4f} | SE: {final_res['SE']:.2f} | SP: {final_res['SP']:.2f} | "
            f"TP {final_res['TP']} TN {final_res['TN']} "
            f"FP {final_res['FP']} FN {final_res['FN']}"
        )
    else:
        print(f"\n⚠️ best_model.pt not found at: {best_path}\n"
              f"    (可能原因：训练期间从未触发 improved 条件，所以没保存 best。)")


if __name__ == "__main__":
    main()
