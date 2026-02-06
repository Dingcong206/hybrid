#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import argparse
from pathlib import Path
from typing import Dict
import math
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
# 2) Dataset：读 tokens.npy（四分类 label 0/1/2/3）
# ============================================================
class TokenNPYDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = False):
        self.csv_path = csv_path
        self.is_train = is_train

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Dataset] CSV 不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)

        # 确保标签是 0,1,2,3
        self.labels = self.df["label"].astype(int).values
        self.class_counts = np.bincount(self.labels, minlength=4)

        print(f"[Dataset] Loaded {len(self.df)} samples from {csv_path} | "
              f"counts(0/1/2/3)={self.class_counts.tolist()} | train={self.is_train}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"])
        x = torch.from_numpy(x).float()
        y = int(self.labels[idx])

        #if self.is_train:
            #x = apply_spec_augment(x, max_mask_t=10, max_mask_f=4, num_masks=2)

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
        mask[i, :T] = False  # False = 非 PAD

    y = torch.stack(ys).view(-1)
    return x_pad, mask, y


# ============================================================
# 4) PatchMix 风格：get_score（two_cls_eval）
#    - SP: normal(0) 命中率
#    - SE: abnormal(1/2/3) 宏平均命中率（注意：two_cls_eval 下 abnormal 命中条件是 pred>0）
# ============================================================
def get_score_patchmix_style(hits, counts):
    eps = 1e-10

    # specificity: class 0
    sp = (hits[0] / (counts[0] + eps)) * 100.0

    # sensitivity: macro avg of classes 1..K-1
    se_list = []
    for c in range(1, len(counts)):
        se_list.append(hits[c] / (counts[c] + eps))
    se = (sum(se_list) / max(1, len(se_list))) * 100.0

    score = (sp + se) / 2.0
    return float(sp), float(se), float(score)


# ============================================================
# 5) 评估：完全对齐 PatchMix validate(two_cls_eval=True)
#    - pred4=argmax
#    - hits/counts 逐类统计（异常类只要 pred>0 就算命中，不要求 pred==label）
#    - SP/SE/Score 用 get_score_patchmix_style 算（异常为宏平均）
#    另外：为了方便你调参，也输出 confusion matrix 的 TP/TN/FP/FN、ACC、F1、PredAbnRate
# ============================================================
@torch.no_grad()
def evaluate_patchmix_two_cls(backbone, classifier, loader, device) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    n_cls = 4
    hits = [0.0] * n_cls
    counts = [0.0] * n_cls

    all_pred4, all_true4 = [], []

    for x, mask, y4 in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y4 = y4.to(device, non_blocking=True)

        feat = backbone(x, mask=mask)
        logits4 = classifier(feat)              # (B,4)
        pred4 = torch.argmax(logits4, dim=1)    # (B,)

        all_pred4.append(pred4.detach().cpu())
        all_true4.append(y4.detach().cpu())

        # PatchMix two_cls_eval 统计
        for i in range(y4.size(0)):
            yt = int(y4[i].item())
            yp = int(pred4[i].item())
            counts[yt] += 1.0

            if yt == 0:
                if yp == 0:
                    hits[0] += 1.0
            else:
                # abnormal：只要预测为 abnormal（>0）就算命中
                if yp > 0:
                    hits[yt] += 1.0

    sp, se, sc = get_score_patchmix_style(hits, counts)

    # 下面这些只是为了打印调参更直观（不影响 PatchMix 的 Score 口径）
    y_pred4 = torch.cat(all_pred4).numpy()
    y_true4 = torch.cat(all_true4).numpy()
    y_pred_bin = (y_pred4 > 0).astype(np.int64)
    y_true_bin = (y_true4 > 0).astype(np.int64)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true_bin, y_pred_bin) * 100.0
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    pred_abn_rate = float((y_pred_bin == 1).mean() * 100.0)

    return {
        "SP": sp,
        "SE": se,
        "Score": sc,
        "ACC": float(acc),
        "F1": float(f1),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "PredAbnRate": float(pred_abn_rate),
        "hits": hits,
        "counts": counts,
    }


# ============================================================
# 6) Seed
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 7) Train (official Train(60%) update params, official Test(40%) eval + pick best)
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str,
                        default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",
                        help="预处理输出目录（包含 train_index.csv / test_index.csv）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="mini-batch size")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="gradient accumulation steps (effective batch = batch_size * accum_steps)")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_routeA_patchmix_eval")
    parser.add_argument("--patience", type=int, default=10)

    # weighted loss：默认关（建议你先稳定跑通）
    parser.add_argument("--use_weighted_loss", action="store_true",
                        help="use class-balanced CE (default OFF)")
    # 想关就不传；想开就加 --use_weighted_loss

    # model args
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=4)

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
        collate_fn=collate_pad, drop_last=False
    )

    dl_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_pad
    )

    print(f"[INFO] train cycles: {len(ds_train)} | test cycles: {len(ds_test)}")
    print(f"[INFO] train class_counts (0/1/2/3): {ds_train.class_counts.tolist()}")

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
        freq = counts / counts.sum()

        # 这里给你一个更温和的权重（比 1/sqrt(freq) 更稳）
        w = 1.0 / (np.power(freq, 0.25) + 1e-12)
        w = w / w.sum() * 4.0

        weight = torch.tensor(w, device=device, dtype=torch.float32)
        print("[INFO] weighted CE weights:", w)
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.0)
    else:
        loss_fn = nn.CrossEntropyLoss()

    accum_steps = args.accum_steps

    params = list(backbone.parameters()) + list(classifier.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))

    updates_per_epoch = math.ceil(len(dl_train) / accum_steps)
    total_steps = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    print("\n🚀 Start training (Train(60%) update params, Test(40%) eval, PatchMix two_cls_eval scoring)\n")

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()

        t0 = time.time()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        for i, (x, mask, y) in enumerate(dl_train):
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                feat = backbone(x, mask=mask)
                logits = classifier(feat)
                loss = loss_fn(logits, y) / accum_steps

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

        # ✅ PatchMix two_cls_eval 口径评估
        test_m = evaluate_patchmix_two_cls(backbone, classifier, dl_test, device)
        test_score = test_m["Score"]

        improved = test_score > best_score + 1e-6
        if improved:
            best_score = test_score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": backbone.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "best_score": best_score,
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
            f"TEST(PatchMixEval) Score {test_m['Score']:.4f} SE {test_m['SE']:.4f} SP {test_m['SP']:.4f} | "
            f"ACC {test_m['ACC']:.4f} F1 {test_m['F1']:.4f} | "
            f"TP {test_m['TP']} TN {test_m['TN']} FP {test_m['FP']} FN {test_m['FN']} | "
            f"PredAbn {test_m['PredAbnRate']:.2f}% | "
            f"{dt:.1f}s"
        )

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] TEST Score 连续 {args.patience} 轮无提升，停止于 epoch {epoch}（best@{best_epoch}）")
            break

    print(f"\n✅ DONE. Best TEST Score={best_score:.4f} @ epoch {best_epoch}")
    print(f"[SAVED] best checkpoint: {ckpt_path}")

    # Final: load best and eval again
    print("\n🚀 Final TEST evaluation (PatchMix two_cls_eval from best checkpoint)\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    classifier.load_state_dict(ckpt["classifier_state"])

    final_m = evaluate_patchmix_two_cls(backbone, classifier, dl_test, device)
    print(
        f"[TEST PatchMixEval] Score {final_m['Score']:.4f} SE {final_m['SE']:.4f} SP {final_m['SP']:.4f} | "
        f"ACC {final_m['ACC']:.4f} F1 {final_m['F1']:.4f} | "
        f"TP {final_m['TP']} TN {final_m['TN']} FP {final_m['FP']} FN {final_m['FN']} | "
        f"PredAbn {final_m['PredAbnRate']:.2f}%"
    )


if __name__ == "__main__":
    main()
