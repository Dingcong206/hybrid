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
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# =========================
# Path / import
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels import build_model


# ============================================================
# 1) SpecAugment（对 tokens 做维度遮挡）
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

        # feature mask
        f_width = random.randint(0, max_mask_f)
        f_start = random.randint(0, max(0, D - f_width))
        if f_width > 0:
            x_aug[:, f_start:f_start + f_width] = 0

    return x_aug


# ============================================================
# 2) Dataset：读 tokens.npy（原始四分类 label 0/1/2/3）
#    训练/评估时内部转二分类 y_bin: 0=normal, 1=abnormal(1/2/3)
# ============================================================
class TokenNPYDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        is_train: bool = False,
        specaug: bool = False,
        max_mask_t: int = 10,
        max_mask_f: int = 4,
        num_masks: int = 2,
    ):
        self.csv_path = csv_path
        self.is_train = is_train
        self.specaug = specaug
        self.max_mask_t = max_mask_t
        self.max_mask_f = max_mask_f
        self.num_masks = num_masks

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Dataset] CSV 不存在: {csv_path}")

        df = pd.read_csv(csv_path).reset_index(drop=True)
        self.df = df

        self.y4 = self.df["label"].astype(int).values
        self.y_bin = (self.y4 > 0).astype(np.int64)

        self.class_counts_4 = np.bincount(self.y4, minlength=4)
        self.class_counts_bin = np.bincount(self.y_bin, minlength=2)

        print(
            f"[Dataset] Loaded {len(self.df)} samples from {csv_path} | "
            f"counts4(0/1/2/3)={self.class_counts_4.tolist()} | "
            f"counts2(N/Abn)={self.class_counts_bin.tolist()} | "
            f"train={self.is_train} specaug={self.specaug}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"])
        x = torch.from_numpy(x).float()

        y4 = int(self.y4[idx])
        yb = 1 if y4 > 0 else 0

        if self.is_train and self.specaug:
            x = apply_spec_augment(
                x, max_mask_t=self.max_mask_t, max_mask_f=self.max_mask_f, num_masks=self.num_masks
            )

        return x, torch.tensor(yb, dtype=torch.long)


# ============================================================
# 3) collate：pad + mask
# ============================================================
def collate_pad(batch):
    """
    batch: List[(x(T,D), y)]
    return:
      x_pad: (B, T_max, D)
      mask : (B, T_max)  True=PAD
      y    : (B,)
    """
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    T_max = max(lens)
    B = len(xs)

    x_pad = torch.zeros(B, T_max, D, dtype=torch.float32)
    mask = torch.ones(B, T_max, dtype=torch.bool)

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        mask[i, :T] = False  # False = 非 PAD

    y = torch.stack(ys).view(-1)
    return x_pad, mask, y


# ============================================================
# 4) ICBHI Score (二分类): SP + SE / 2
#    SE = TP/(TP+FN), SP = TN/(TN+FP)
# ============================================================
def icbhi_score_from_cm(tn, fp, fn, tp) -> Tuple[float, float, float]:
    eps = 1e-10
    sp = 100.0 * (tn / (tn + fp + eps))
    se = 100.0 * (tp / (tp + fn + eps))
    score = (sp + se) / 2.0
    return float(sp), float(se), float(score)


@torch.no_grad()
def evaluate_binary_icbhi(
    backbone,
    classifier,
    loader,
    device,
    thr: float = 0.5,
) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    all_prob, all_true = [], []

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = backbone(x, mask=mask)
        logit = classifier(feat).view(-1)   # (B,)
        prob = torch.sigmoid(logit)         # P(abnormal)
        all_prob.append(prob.detach().cpu())
        all_true.append(y.detach().cpu())

    p = torch.cat(all_prob).numpy()
    t = torch.cat(all_true).numpy().astype(np.int64)
    pred = (p >= thr).astype(np.int64)

    cm = confusion_matrix(t, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp, se, score = icbhi_score_from_cm(tn, fp, fn, tp)
    acc = accuracy_score(t, pred) * 100.0
    f1 = f1_score(t, pred, zero_division=0)

    pred_abn_rate = float((pred == 1).mean() * 100.0)

    return {
        "THR": float(thr),
        "ICBHI": float(score),
        "SP": float(sp),
        "SE": float(se),
        "ACC": float(acc),
        "F1": float(f1),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "PredAbnRate": float(pred_abn_rate),
    }


@torch.no_grad()
def find_best_threshold_on_val(
    backbone,
    classifier,
    val_loader,
    device,
    thr_list: List[float],
) -> Dict[str, float]:
    best = None
    for thr in thr_list:
        m = evaluate_binary_icbhi(backbone, classifier, val_loader, device, thr=thr)
        if (best is None) or (m["ICBHI"] > best["ICBHI"] + 1e-9):
            best = m
    return best


# ============================================================
# 5) Seed
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 6) Train
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str,
                        default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",
                        help="预处理输出目录（包含 train_index.csv / test_index.csv）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_icbhi_binary_score_fixed")
    parser.add_argument("--patience", type=int, default=20)

    # ====== val split ======
    parser.add_argument("--val_ratio", type=float, default=0.1, help="split from train(60%) into val")
    parser.add_argument("--early_metric", type=str, default="ICBHI", choices=["ICBHI"], help="early stop metric")

    # ====== SpecAugment ======
    parser.add_argument("--specaug", action="store_true", help="enable SpecAugment on tokens (default OFF)")
    parser.add_argument("--max_mask_t", type=int, default=10)
    parser.add_argument("--max_mask_f", type=int, default=4)
    parser.add_argument("--num_masks", type=int, default=2)

    # ====== model args ======
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_len", type=int, default=1024)

    # ====== amp ======
    parser.add_argument("--amp", action="store_true", help="use mixed precision (cuda only)")

    # ====== weighted loss (binary) ======
    parser.add_argument("--use_pos_weight", action="store_true", help="use pos_weight for BCE (default OFF)")

    # ====== threshold scanning ======
    parser.add_argument("--thr_min", type=float, default=0.1)
    parser.add_argument("--thr_max", type=float, default=0.9)
    parser.add_argument("--thr_step", type=float, default=0.05)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] project_root: {PROJECT_ROOT}")
    if device.type == "cuda":
        print("[DEBUG] device_count:", torch.cuda.device_count())
        print("[DEBUG] current_device:", torch.cuda.current_device())
        print("[DEBUG] device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pt")

    root = Path(args.root)
    train_csv = root / "train_index.csv"
    test_csv = root / "test_index.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"找不到：\n{train_csv}\n{test_csv}")

    # ===== load train dataset once for split =====
    full_train = TokenNPYDataset(
        str(train_csv),
        is_train=True,
        specaug=args.specaug,
        max_mask_t=args.max_mask_t,
        max_mask_f=args.max_mask_f,
        num_masks=args.num_masks,
    )
    test_ds = TokenNPYDataset(str(test_csv), is_train=False)

    # ===== stratified split on binary labels =====
    idx = np.arange(len(full_train))
    ybin = full_train.y_bin
    val_ratio = float(args.val_ratio)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.seed)
    train_idx, val_idx = next(sss.split(idx, ybin))

    train_ds = Subset(full_train, train_idx.tolist())
    # val 不做 specaug：用一个不开增强的 dataset wrapper（最稳）
    val_base = TokenNPYDataset(str(train_csv), is_train=False)
    val_ds = Subset(val_base, val_idx.tolist())

    print(f"[INFO] split train={len(train_ds)} val={len(val_ds)} (from train_index.csv={len(full_train)})")
    print(f"[INFO] test cycles={len(test_ds)} (from test_index.csv)")

    dl_train = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pad, drop_last=False
    )
    dl_val = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pad, drop_last=False
    )
    dl_test = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pad, drop_last=False
    )

    # ===== build backbone =====
    backbone = build_model(
        in_dim=args.in_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    if not hasattr(backbone, "final_feat_dim"):
        raise RuntimeError("build_model 返回对象没有 final_feat_dim。")

    # ===== binary classifier head: 1 logit =====
    classifier = nn.Linear(backbone.final_feat_dim, 1).to(device)

    # ===== loss =====
    # y is 0/1, use BCEWithLogitsLoss
    if args.use_pos_weight:
        # estimate pos_weight from full_train binary counts (on train_index.csv)
        n0, n1 = full_train.class_counts_bin.tolist()  # [normal, abnormal]
        # pos_weight = Nneg / Npos
        pos_weight = torch.tensor([float(n0) / max(1.0, float(n1))], device=device)
        print(f"[INFO] BCE pos_weight: {pos_weight.item():.4f}  (Nneg={n0}, Npos={n1})")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    params = list(backbone.parameters()) + list(classifier.parameters())
    optim = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )

    # ===== AMP (cuda only) =====
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== scheduler aligned to TRUE optimizer steps =====
    accum_steps = int(args.accum_steps)
    opt_steps_per_epoch = math.ceil(len(dl_train) / accum_steps)
    total_opt_steps = opt_steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_opt_steps)

    thr_list = []
    t = args.thr_min
    while t <= args.thr_max + 1e-12:
        thr_list.append(round(float(t), 6))
        t += args.thr_step
    print(f"[INFO] threshold candidates: {thr_list}")

    best_val = -1.0
    best_epoch = -1
    best_thr = 0.5
    bad_epochs = 0

    print("\n🚀 Start training (Binary train, Val early-stop by ICBHI=(SE+SP)/2, threshold scanning on VAL)\n")

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()

        t0 = time.time()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        for i, (x, mask, y) in enumerate(dl_train):
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()  # BCE needs float

            with torch.autocast(device_type=device.type, enabled=use_amp):
                feat = backbone(x, mask=mask)
                logit = classifier(feat).view(-1)  # (B,)
                loss = loss_fn(logit, y) / accum_steps

            scaler.scale(loss).backward()
            running += float(loss.item() * accum_steps)

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 5.0)

                scaler.step(optim)
                scaler.update()
                scheduler.step()

                optim.zero_grad(set_to_none=True)

        # tail
        if len(dl_train) % accum_steps != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 5.0)

            scaler.step(optim)
            scaler.update()
            scheduler.step()

            optim.zero_grad(set_to_none=True)

        train_loss = running / max(1, len(dl_train))

        # ===== VAL: scan thresholds, pick best ICBHI =====
        val_best = find_best_threshold_on_val(backbone, classifier, dl_val, device, thr_list)
        val_icbhi = val_best["ICBHI"]

        improved = val_icbhi > best_val + 1e-9
        if improved:
            best_val = val_icbhi
            best_epoch = epoch
            best_thr = float(val_best["THR"])
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": backbone.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "best_val_icbhi": best_val,
                    "best_thr": best_thr,
                    "args": vars(args),
                },
                ckpt_path
            )
            star = "⭐"
        else:
            bad_epochs += 1
            star = " "

        dt = time.time() - t0
        print(
            f"{star} Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | "
            f"VAL ICBHI {val_best['ICBHI']:.4f} SE {val_best['SE']:.4f} SP {val_best['SP']:.4f} "
            f"(thr={val_best['THR']:.2f}) | "
            f"ACC {val_best['ACC']:.4f} F1 {val_best['F1']:.4f} | "
            f"TP {val_best['TP']} TN {val_best['TN']} FP {val_best['FP']} FN {val_best['FN']} | "
            f"PredAbn {val_best['PredAbnRate']:.2f}% | "
            f"{dt:.1f}s"
        )

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] VAL ICBHI 连续 {args.patience} 轮无提升，停止于 epoch {epoch}（best@{best_epoch} thr={best_thr:.2f}）")
            break

    print(f"\n✅ DONE. Best VAL ICBHI={best_val:.4f} @ epoch {best_epoch} (thr={best_thr:.2f})")
    print(f"[SAVED] best checkpoint: {ckpt_path}")

    # ===== Final: load best and test =====
    print("\n🚀 Final TEST evaluation (ICBHI Score=(SE+SP)/2) from best checkpoint\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    classifier.load_state_dict(ckpt["classifier_state"])
    best_thr = float(ckpt.get("best_thr", 0.5))

    test_m = evaluate_binary_icbhi(backbone, classifier, dl_test, device, thr=best_thr)
    print(
        f"[TEST] ICBHI {test_m['ICBHI']:.4f} SE {test_m['SE']:.4f} SP {test_m['SP']:.4f} "
        f"(thr={test_m['THR']:.2f}) | "
        f"ACC {test_m['ACC']:.4f} F1 {test_m['F1']:.4f} | "
        f"TP {test_m['TP']} TN {test_m['TN']} FP {test_m['FP']} FN {test_m['FN']} | "
        f"PredAbn {test_m['PredAbnRate']:.2f}%"
    )


if __name__ == "__main__":
    main()
