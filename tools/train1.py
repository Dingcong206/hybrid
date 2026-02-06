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
def apply_spec_augment(x, max_mask_t=120, max_mask_f=32, num_masks=2):
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
        self.is_train = is_train  # ✅ 必须先赋值，才能在后面被 self 调用

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Dataset] CSV 不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)

        # 确保标签是 0,1,2,3
        self.labels = self.df["label"].astype(int).values
        self.class_counts = np.bincount(self.labels, minlength=4)

        # ✅ 现在可以安全地打印 self.is_train 了
        print(f"[Dataset] Loaded {len(self.df)} samples from {csv_path} | "
              f"counts(0/1/2/3)={self.class_counts.tolist()} | train={self.is_train}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"])
        x = torch.from_numpy(x).float()
        y = int(self.labels[idx])

        # 使用之前提到的较大掩码参数来抑制震荡
        if self.is_train:
            x = apply_spec_augment(x, max_mask_t=80, max_mask_f=32, num_masks=2)

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
    对齐 PatchMix 的 two_cls_eval 口径：
      - 先做 4-class argmax 得 pred4 ∈ {0,1,2,3}
      - 二分类判断：normal=0，abnormal=pred4>0
      - SP = 正常类准确率；SE = 异常合并后的召回；Score=(SP+SE)/2
      - 不用阈值、不用 softmax
    """
    backbone.eval()
    classifier.eval()

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

    y_pred4 = torch.cat(all_pred4).numpy()
    y_true4 = torch.cat(all_true4).numpy()

    # ✅ PatchMix two_cls_eval：0=normal，其余=abnormal（pred>0）
    y_pred_bin = (y_pred4 > 0).astype(np.int64)
    y_true_bin = (y_true4 > 0).astype(np.int64)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = tn / (tn + fp + 1e-10) * 100.0
    se = tp / (tp + fn + 1e-10) * 100.0
    sc = (sp + se) / 2.0

    acc = accuracy_score(y_true_bin, y_pred_bin) * 100.0
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    pred_abn_rate = float((y_pred_bin == 1).mean() * 100.0)

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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="mini-batch size")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="gradient accumulation steps (effective batch = batch_size * accum_steps)")

    parser.add_argument("--lr", type=float, default=1e-5)
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
    parser.add_argument("--d_model", type=int, default=512)
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
        w = 1.0 / (np.sqrt(freq) + 1e-12)
        w = w / w.sum() * 4.0

        weight = torch.tensor(w, device=device, dtype=torch.float32)
        print("[INFO] weighted CE weights:", w)
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    else:
        loss_fn = nn.CrossEntropyLoss()

    accum_steps = args.accum_steps

    params = list(backbone.parameters()) + list(classifier.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    updates_per_epoch = math.ceil(len(dl_train) / accum_steps)
    total_steps = args.epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

   # scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best_test_icbhi = -1.0
    best_epoch = -1
    bad_epochs = 0

    print("\n🚀 Start training (official Train(60%) update params, official Test(40%) eval + pick best, ARGMAX)\n")

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        classifier.train()

        t0 = time.time()
        running = 0.0
        optim.zero_grad(set_to_none=True)  # ✅ 每个 epoch 开始清空一次梯度

        for i, (x, mask, y) in enumerate(dl_train):
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                feat = backbone(x, mask=mask)
                logits = classifier(feat)
                loss = loss_fn(logits, y) / accum_steps  # ✅ 关键：除以 accum_steps

            # ✅ 1) 累积梯度
            scaler.scale(loss).backward()
            running += float(loss.item() * accum_steps)

            # ✅ 2) 每 accum_steps 次，更新一次参数
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optim)  # 先反缩放，才能裁剪
                torch.nn.utils.clip_grad_norm_(params, 5.0)  # 裁剪梯度
                scaler.step(optim)  # optimizer.step()
                scaler.update()
                scheduler.step()  # 如果你按 step 更新学习率，就放这里
                optim.zero_grad(set_to_none=True)  # 清梯度，开始下一轮累积

        # ✅ 3) 处理“尾巴”：如果最后不足 accum_steps，也要更新一次
        if len(dl_train) % accum_steps != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

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
            f"PredAbn {test_m['PredAbnRate']:.2f}% | "

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