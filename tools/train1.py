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
# 1) SpecAugment（对 tokens 做维度遮挡）——先弱增强求稳定
# ============================================================
def apply_spec_augment(x, max_mask_t=10, max_mask_f=3, num_masks=1):
    T, D = x.shape
    x_aug = x.clone()
    for _ in range(num_masks):
        t_width = random.randint(0, max_mask_t)
        t_start = random.randint(0, max(0, T - t_width))
        if t_width > 0:
            x_aug[t_start:t_start + t_width, :] = 0

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
        self.binary_labels = np.array([0 if l == 0 else 1 for l in raw_labels], dtype=np.int64)
        self.class_counts = np.bincount(self.binary_labels, minlength=2)

        u = np.unique(self.binary_labels)
        assert set(u.tolist()).issubset({0, 1}), f"[Dataset BUG] binary_labels not in {{0,1}}: {u}"

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
            x = apply_spec_augment(x)

        return x, torch.tensor(y, dtype=torch.long)


# ============================================================
# 3) collate：pad + mask
# ============================================================
def collate_pad(batch):
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

    valid_lens = (~mask).sum(dim=1)
    assert torch.all(valid_lens > 0), "[COLLATE BUG] some sample has 0 valid tokens"

    y = torch.stack(ys).view(-1)
    return x_pad, mask, y


# ============================================================
# 4) Eval: argmax
# ============================================================
@torch.no_grad()
def evaluate_argmax(backbone, classifier, loader, device) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    all_pred, all_true = [], []
    abnormal_rates = []

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = backbone(x, mask=mask)
        logits = classifier(feat)  # (B,2)
        pred = torch.argmax(logits, dim=1)

        abnormal_rates.append(float((pred == 1).float().mean().item()))
        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = tn / (tn + fp + 1e-10) * 100.0
    se = tp / (tp + fn + 1e-10) * 100.0
    sc = (sp + se) / 2.0

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pred_abn_rate = float(np.mean(abnormal_rates) * 100.0)

    return {
        "SP": float(sp),
        "SE": float(se),
        "Score": float(sc),
        "ACC": float(acc),
        "F1": float(f1),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "PredAbnRate": float(pred_abn_rate),
    }


# ============================================================
# 5) Eval: threshold sweep（只做参考打印，不用于保存/早停）
# ============================================================
@torch.no_grad()
def evaluate_sweep_threshold(backbone, classifier, loader, device, thr_list=None) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    if thr_list is None:
        thr_list = np.linspace(0.05, 0.95, 19)

    all_prob, all_true = [], []
    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = backbone(x, mask=mask)
        logits = classifier(feat)  # (B,2)
        prob_abn = torch.softmax(logits.float(), dim=1)[:, 1]
        all_prob.append(prob_abn.detach().cpu())
        all_true.append(y.detach().cpu())

    prob = torch.cat(all_prob).numpy()
    y_true = torch.cat(all_true).numpy()

    best = {
        "BestThr": 0.5, "Score": -1.0,
        "SE": 0.0, "SP": 0.0, "ACC": 0.0, "F1": 0.0,
        "TP": 0, "TN": 0, "FP": 0, "FN": 0,
        "PredAbnRate": 0.0
    }

    for thr in thr_list:
        y_pred = (prob >= thr).astype(np.int64)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sp = tn / (tn + fp + 1e-10) * 100.0
        se = tp / (tp + fn + 1e-10) * 100.0
        score = (sp + se) / 2.0

        acc = accuracy_score(y_true, y_pred) * 100.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pred_abn_rate = float((y_pred == 1).mean() * 100.0)

        if score > best["Score"]:
            best.update({
                "BestThr": float(thr),
                "Score": float(score),
                "SE": float(se),
                "SP": float(sp),
                "ACC": float(acc),
                "F1": float(f1),
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
                "PredAbnRate": float(pred_abn_rate),
            })

    return best


# ============================================================
# 6) Eval: fixed threshold（用 ckpt 保存的 best_thr）
# ============================================================
@torch.no_grad()
def evaluate_fixed_threshold(backbone, classifier, loader, device, thr=0.5) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    all_pred, all_true = [], []
    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = backbone(x, mask=mask)
        logits = classifier(feat)  # (B,2)
        prob_abn = torch.softmax(logits.float(), dim=1)[:, 1]
        pred = (prob_abn >= thr).long()

        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = tn / (tn + fp + 1e-10) * 100.0
    se = tp / (tp + fn + 1e-10) * 100.0
    score = (sp + se) / 2.0

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pred_abn_rate = float((y_pred == 1).mean() * 100.0)

    return {
        "Thr": float(thr),
        "SP": float(sp), "SE": float(se), "Score": float(score),
        "ACC": float(acc), "F1": float(f1),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "PredAbnRate": float(pred_abn_rate),
    }


# ============================================================
# 7) Seed
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 8) Train (✅ ARGMAX 保存 + 早停)
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_routeA_argmax_best")
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--use_weighted_loss", action="store_true", default=False)
    parser.add_argument("--no_weighted_loss", action="store_false", dest="use_weighted_loss")

    # model args
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=2)

    # amp
    parser.add_argument("--amp", action="store_true", default=True)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] project_root: {PROJECT_ROOT}")
    if device.type == "cuda":
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
    params = list(backbone.parameters()) + list(classifier.parameters())

    # loss
    if args.use_weighted_loss:
        counts = ds_train.class_counts.astype(np.float32)
        freq = counts / counts.sum()
        w = 1.0 / (np.sqrt(freq) + 1e-12)
        w = w / w.sum() * 2.0
        weight = torch.tensor(w, device=device, dtype=torch.float32)
        print("[INFO] CE weights (sqrt):", w, "| w1/w0=", float(w[1] / (w[0] + 1e-12)))
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))

    # warmup + cosine, step 以“参数更新次数”为单位
    steps_per_epoch = math.ceil(len(dl_train) / args.accum_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # ✅ ARGMAX 保存/早停
    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    print("\n🚀 Start training (SAVE/EARLYSTOP by ARGMAX; thr-sweep only for logging)\n")

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

            with torch.amp.autocast("cuda", enabled=args.amp):
                feat = backbone(x, mask=mask)
                logits = classifier(feat)
                loss = loss_fn(logits, y) / args.accum_steps

            scaler.scale(loss).backward()
            running += float(loss.item() * args.accum_steps)

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(dl_train):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 1.0)

                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

        train_loss = running / max(1, len(dl_train))

        # eval
        test_arg = evaluate_argmax(backbone, classifier, dl_test, device)
        test_best = evaluate_sweep_threshold(backbone, classifier, dl_test, device)  # 仅打印参考

        score_arg = test_arg["Score"]

        # ✅ 关键：保存/早停用 ARGMAX
        improved = score_arg > best_score + 1e-6
        if improved:
            best_score = score_arg
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": backbone.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "best_score_argmax": float(best_score),
                    "best_thr_log": float(test_best["BestThr"]),  # 仅记录
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
            f"ARGMAX Score {test_arg['Score']:.4f} SE {test_arg['SE']:.2f} SP {test_arg['SP']:.2f} "
            f"PredAbn {test_arg['PredAbnRate']:.2f}% || "
            f"(log) BestThr {test_best['BestThr']:.2f} Score {test_best['Score']:.4f} "
            f"SE {test_best['SE']:.2f} SP {test_best['SP']:.2f} PredAbn {test_best['PredAbnRate']:.2f}% | "
            f"{dt:.1f}s"
        )

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] ARGMAX Score 连续 {args.patience} 轮无提升，停止于 epoch {epoch}（best@{best_epoch}）")
            break

    print(f"\n✅ DONE. Best(ARGMAX) TEST Score={best_score:.4f} @ epoch {best_epoch}")
    print(f"[SAVED] best checkpoint: {ckpt_path}")

    # final eval from best checkpoint
    print("\n🚀 Final TEST evaluation (from best checkpoint)\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    classifier.load_state_dict(ckpt["classifier_state"])

    final_arg = evaluate_argmax(backbone, classifier, dl_test, device)

    # 如果你仍想看“最终扫阈值”的上限，只做展示（注意：这属于 test 上调参，不要当最终论文结论）
    final_best = evaluate_sweep_threshold(backbone, classifier, dl_test, device)

    print(
        f"[FINAL argmax] Score {final_arg['Score']:.4f} SE {final_arg['SE']:.2f} SP {final_arg['SP']:.2f} "
        f"PredAbn {final_arg['PredAbnRate']:.2f}%"
    )
    print(
        f"[FINAL (log) bestthr] BestThr {final_best['BestThr']:.2f} Score {final_best['Score']:.4f} "
        f"SE {final_best['SE']:.2f} SP {final_best['SP']:.2f} PredAbn {final_best['PredAbnRate']:.2f}%"
    )


if __name__ == "__main__":
    main()
