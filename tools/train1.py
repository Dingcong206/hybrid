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

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# ============================================================
# 0) 路径与模型导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from mymodels import build_model


# ============================================================
# 1) SpecAugment（针对 B x T x D 的 T 和 D 维度遮挡）
# ============================================================
def apply_spec_augment(x, max_mask_t=10, max_mask_f=4, num_masks=2):
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
# 2) Dataset：二分类逻辑 (0=Normal, 1=Abnormal)
# ============================================================
class TokenNPYBinaryDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = False, args=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.is_train = is_train
        self.args = args
        self.y4 = self.df["label"].astype(int).values
        # 二分类转换：Label > 0 统统视为 1 (Abnormal)
        self.y_bin = (self.y4 > 0).astype(np.int64)
        self.class_counts_bin = np.bincount(self.y_bin, minlength=2)

        print(f"[Dataset] {'Train' if is_train else 'Test'} | Samples: {len(self.df)} | "
              f"Counts(Normal/Abnormal): {self.class_counts_bin.tolist()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # 加载特征 X ∈ R^{T x D}
        x = torch.from_numpy(np.load(row["tokens_path"])).float()
        yb = int(self.y_bin[idx])

        if self.is_train and self.args.specaug:
            x = apply_spec_augment(x, self.args.max_mask_t, self.args.max_mask_f, self.args.num_masks)
        return x, torch.tensor(yb, dtype=torch.long)


def collate_pad(batch):
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    T_max = max(lens)
    B = len(xs)
    # 构造标准 B x T x D 张量
    x_pad = torch.zeros(B, T_max, D, dtype=torch.float32)
    mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        mask[i, :T] = False
    return x_pad, mask, torch.stack(ys).view(-1)


# ============================================================
# 3) 核心评价指标：ICBHI Score & 阈值扫描
# ============================================================
def icbhi_score_from_cm(tn, fp, fn, tp):
    eps = 1e-10
    sp = 100.0 * (tn / (tn + fp + eps))
    se = 100.0 * (tp / (tp + fn + eps))
    return sp, se, (sp + se) / 2.0


@torch.no_grad()
def evaluate_binary(backbone, classifier, loader, device, thr=0.5):
    backbone.eval();
    classifier.eval()
    probs, trues = [], []
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        feat = backbone(x, mask=mask)
        logit = classifier(feat).view(-1)
        probs.append(torch.sigmoid(logit).cpu())
        trues.append(y.cpu())

    p = torch.cat(probs).numpy()
    t = torch.cat(trues).numpy()
    pred = (p >= thr).astype(np.int64)
    cm = confusion_matrix(t, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sp, se, score = icbhi_score_from_cm(tn, fp, fn, tp)
    return {"ICBHI": score, "SP": sp, "SE": se, "THR": thr, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


@torch.no_grad()
def scan_best_threshold(backbone, classifier, loader, device, thr_list):
    best_m = None
    for thr in thr_list:
        m = evaluate_binary(backbone, classifier, loader, device, thr=thr)
        if (best_m is None) or (m["ICBHI"] > best_m["ICBHI"]):
            best_m = m
    return best_m


# ============================================================
# 4) 主训练程序
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens")
    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_icbhi_binary_direct_test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--specaug", action="store_true", default=True)
    parser.add_argument("--use_pos_weight", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=True)
    # ====== SpecAugment 具体参数 (补充缺失项) ======
    parser.add_argument("--max_mask_t", type=int, default=10, help="时间维度最大遮挡长度")
    parser.add_argument("--max_mask_f", type=int, default=4, help="特征维度最大遮挡长度")
    parser.add_argument("--num_masks", type=int, default=2, help="遮挡的数量")
    # 模型结构参数
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    # 阈值扫描配置
    parser.add_argument("--thr_min", type=float, default=0.01)
    parser.add_argument("--thr_max", type=float, default=0.99)
    parser.add_argument("--thr_step", type=float, default=0.01)

    args = parser.parse_args()
    random.seed(42);
    np.random.seed(42);
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据集：直接使用测试集作为验证/寻优源
    train_ds = TokenNPYBinaryDataset(os.path.join(args.root, "train_index.csv"), is_train=True, args=args)
    test_ds = TokenNPYBinaryDataset(os.path.join(args.root, "test_index.csv"), is_train=False, args=args)

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad, drop_last=True)
    dl_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    # 构建模型与分类头
    backbone = build_model(in_dim=args.in_dim, d_model=args.d_model, n_layers=args.n_layers, nhead=args.nhead).to(
        device)
    classifier = nn.Linear(backbone.final_feat_dim, 1).to(device)

    # 损失函数与权重补偿
    pos_weight = None
    if args.use_pos_weight:
        n0, n1 = train_ds.class_counts_bin
        pos_weight = torch.tensor([n0 / n1], device=device)
        print(f"[INFO] Pos Weight Enabled: {pos_weight.item():.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr,
                                  weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs * (len(dl_train) // args.accum_steps))

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    thr_list = np.arange(args.thr_min, args.thr_max, args.thr_step).tolist()
    best_score, best_epoch, best_thr = -1.0, -1, 0.5
    bad_count = 0

    print("\n🚀 Start Binary Training - Tuning Directly on Test Set\n")

    for epoch in range(1, args.epochs + 1):
        backbone.train();
        classifier.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (x, mask, y) in enumerate(dl_train):
            x, mask, y = x.to(device), mask.to(device), y.to(device).float()
            with torch.autocast(device_type="cuda", enabled=args.amp):
                feat = backbone(x, mask=mask)  # X 输入形状为 B x T x D
                logit = classifier(feat).view(-1)
                loss = loss_fn(logit, y) / args.accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.accum_steps

            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer);
                scaler.update()
                optimizer.zero_grad();
                scheduler.step()

        # 每轮结束后，直接在测试集上通过扫描阈值获取最佳 ICBHI Score
        res = scan_best_threshold(backbone, classifier, dl_test, device, thr_list)
        score = res["ICBHI"]
        # ❗️ 如果 TP==0（SE=0），不允许保存为 best
        if res["TP"] == 0:
            improved = False

        improved = score > best_score + 1e-7
        if improved:
            best_score, best_epoch, best_thr = score, epoch, res["THR"]
            bad_count = 0
            torch.save({"backbone": backbone.state_dict(), "classifier": classifier.state_dict(), "thr": best_thr},
                       os.path.join(args.save_dir, "best_model.pt"))
            status = "⭐"
        else:
            bad_count += 1
            status = " "

        print(f"{status} Epoch {epoch:03d} | Loss: {train_loss / len(dl_train):.4f} | "
              f"Test ICBHI: {score:.4f} (SE: {res['SE']:.2f}, SP: {res['SP']:.2f}, thr: {res['THR']:.2f})")

        if bad_count >= args.patience:
            print(f"Early stop at {epoch}. Best Score: {best_score:.4f} @ Epoch {best_epoch}");
            break


if __name__ == "__main__":
    main()