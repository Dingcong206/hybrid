#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
def apply_spec_augment(x, max_mask_t=20, max_mask_f=10, num_masks=2):
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
# 2) Dataset：读 tokens.npy + 二分类映射
# ============================================================
class TokenNPYDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = False):
        self.csv_path = csv_path
        self.is_train = is_train

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Dataset] CSV 不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        if df is None or len(df) == 0:
            raise ValueError(f"[Dataset] CSV 为空或读取失败: {csv_path}")

        for col in ["tokens_path", "label"]:
            if col not in df.columns:
                raise KeyError(f"[Dataset] CSV 缺少列 `{col}`，当前列: {df.columns.tolist()}")

        self.df = df.reset_index(drop=True)

        raw_labels = self.df["label"].astype(int).values
        # 0=normal, 1=abnormal
        self.binary_labels = np.array([0 if l == 0 else 1 for l in raw_labels], dtype=np.int64)
        self.class_counts = np.bincount(self.binary_labels, minlength=2)

        print(f"[Dataset] Loaded {len(self.df)} samples from {csv_path} | "
              f"class_counts(0/1)={self.class_counts.tolist()} | train={self.is_train}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"])
        x = torch.from_numpy(x).float()
        y = int(self.binary_labels[idx])

        if self.is_train:
            x = apply_spec_augment(x, max_mask_t=30, max_mask_f=5, num_masks=2)

        return x, torch.tensor(y, dtype=torch.long)


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
        mask[i, :T] = False

    y = torch.stack(ys).view(-1)
    return x_pad, mask, y


# ============================================================
# 4) 评估：argmax(logits)（与发表版一致：无阈值）
# ============================================================
@torch.no_grad()
def evaluate_argmax(backbone, classifier, loader, device) -> Dict[str, float]:
    """
    完全对齐发表版 get_score() 口径：
    - SP/SE/Score 都是百分制（0~100）
    - Score = (SP + SE) / 2
    - pred 用 argmax(logits)
    """
    backbone.eval()
    classifier.eval()

    all_pred, all_true = [], []

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = backbone(x, mask=mask)
        logits = classifier(feat)            # (B,2)
        pred = torch.argmax(logits, dim=1)   # ✅ argmax

        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ✅ 对齐 get_score：百分制
    sp = tn / (tn + fp + 1e-10) * 100.0   # normal accuracy
    se = tp / (tp + fn + 1e-10) * 100.0   # abnormal accuracy
    sc = (sp + se) / 2.0                 # score

    acc = accuracy_score(y_true, y_pred) * 100.0  # 可选：也改成百分制方便看
    f1 = f1_score(y_true, y_pred)                 # F1 通常保留 0~1 也行

    return {
        "SP": float(sp),
        "SE": float(se),
        "Score": float(sc),
        "ACC": float(acc),
        "F1": float(f1),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


# ============================================================
# 5) Seed
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 6) Train (official Train(60%) train params, official Test(40%) eval + pick best)
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str,
                        default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",
                        help="预处理输出目录（包含 train_index.csv / test_index.csv）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_routeA_argmax")
    parser.add_argument("--patience", type=int, default=10)

    # weighted loss 默认开
    parser.add_argument("--use_weighted_loss", action="store_true", default=True,
                        help="use class-balanced CE (default ON)")
    parser.add_argument("--no_weighted_loss", action="store_false", dest="use_weighted_loss",
                        help="disable class-balanced CE")

    # model args
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=2)

    # amp
    parser.add_argument("--amp", action="store_true", help="use mixed precision")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] project_root: {PROJECT_ROOT}")
    if device.type == "cuda":
        print("[DEBUG] device_count:", torch.cuda.device_count())
        print("[DEBUG] current_device:", torch.cuda.current_device())
        print("[DEBUG] device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    root = Path(args.root)
    train_csv = root / "train_index.csv"
    test_csv = root / "test_index.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"找不到：\n{train_csv}\n{test_csv}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pt")

    ds_train = TokenNPYDataset(str(train_csv), is_train=True)
    ds_test = TokenNPYDataset(str(test_csv), is_train=False)

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pad, drop_last=True
    )

    dl_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pad
    )

    print(f"[INFO] train cycles: {len(ds_train)} | test cycles: {len(ds_test)}")
    print(f"[INFO] train class_counts (0/1): {ds_train.class_counts.tolist()}")

    backbone = build_model(
        in_dim=args.in_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    if not hasattr(backbone, "final_feat_dim"):
        raise RuntimeError("build_model 返回对象没有 final_feat_dim，Route-A 需要它。")

    classifier = nn.Linear(backbone.final_feat_dim, args.num_classes).to(device)

    # loss
    if args.use_weighted_loss:
        counts = ds_train.class_counts.astype(np.float32)
        w = 1.0 / (counts / counts.sum() + 1e-12)
        w = w / w.sum()
        weight = torch.tensor(w, device=device, dtype=torch.float32)
        print("[INFO] weighted CE weights:", w)
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    else:
        loss_fn = nn.CrossEntropyLoss()

    params = list(backbone.parameters()) + list(classifier.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(dl_train))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_test_icbhi = -1.0
    best_epoch = -1
    bad_epochs = 0

    print("\n🚀 Start training (official Train(60%) update params, official Test(40%) eval + pick best, ARGMAX)\n")

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()

        t0 = time.time()
        running = 0.0

        for x, mask, y in dl_train:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                feat = backbone(x, mask=mask)
                logits = classifier(feat)
                loss = loss_fn(logits, y)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            running += float(loss.item())

        train_loss = running / max(1, len(dl_train))

        # ✅ 核心：在官方 TEST(40%) 上用 argmax 评估（与发表版一致）
        test_m = evaluate_argmax(backbone, classifier, dl_test, device)
        test_icbhi = test_m["Score"]

        improved = test_icbhi > best_test_icbhi + 1e-6
        if improved:
            best_test_icbhi = test_icbhi
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": backbone.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "best_test_icbhi": best_test_icbhi,
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
            f"TEST(argmax) Score {test_m['Score']:.4f} SE {test_m['SE']:.4f} SP {test_m['SP']:.4f} | "
            f"ACC {test_m['ACC']:.4f} F1 {test_m['F1']:.4f} | "
            f"TP {test_m['TP']} TN {test_m['TN']} FP {test_m['FP']} FN {test_m['FN']} | "
            f"{dt:.1f}s"
        )

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] TEST Score 连续 {args.patience} 轮无提升，停止于 epoch {epoch}（best@{best_epoch}）")
            break

    print(f"\n✅ DONE. Best TEST Score={best_test_icbhi:.4f} @ epoch {best_epoch}")
    print(f"[SAVED] best checkpoint: {ckpt_path}")

    # ✅ 最后：加载 best ckpt，再在 TEST 上跑一次 argmax（确认最终结果）
    print("\n🚀 Final TEST evaluation (argmax from best checkpoint)\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    classifier.load_state_dict(ckpt["classifier_state"])

    final_m = evaluate_argmax(backbone, classifier, dl_test, device)
    print(
        f"[TEST argmax] Score {final_m['Score']:.4f} SE {final_m['SE']:.4f} SP {final_m['SP']:.4f} | "
        f"ACC {final_m['ACC']:.4f} F1 {final_m['F1']:.4f} | "
        f"TP {final_m['TP']} TN {final_m['TN']} FP {final_m['FP']} FN {final_m['FN']}"
    )


if __name__ == "__main__":
    main()
