#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# ============================================================
# 0) 路径设置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import build_model


# ============================================================
# 1) Dataset
# ============================================================
class TokenNPYDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        labels = self.df["label"].astype(int).values
        self.class_counts = np.bincount(labels, minlength=4)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"])
        return torch.from_numpy(x).float(), torch.tensor(int(row["label"]), dtype=torch.long)


def collate_pad(batch):
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    T_max = max(lens)
    B = len(xs)
    x_pad = torch.zeros(B, T_max, D)
    mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        mask[i, :T] = False
    return x_pad, mask, torch.stack(ys).view(-1)


# ============================================================
# 2) Evaluate
# ============================================================
@torch.no_grad()
def evaluate_icbhi(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_pred4, all_true4 = [], []

    for x, mask, y in loader:
        x, mask = x.to(device), mask.to(device)
        # 解包：忽略 token_logits
        logits, _ = model(x, mask=mask)

        # 确保在评估时也是 float32 比较稳
        pred4 = torch.argmax(logits.float(), dim=1) if logits.shape[1] > 1 else (torch.sigmoid(logits) > 0.5).long()

        all_pred4.append(pred4.cpu())
        all_true4.append(y.cpu())

    y_pred4 = torch.cat(all_pred4).numpy()
    y_true4 = torch.cat(all_true4).numpy()

    y_pred2 = (y_pred4 != 0).astype(np.int64)
    y_true2 = (y_true4 != 0).astype(np.int64)

    cm = confusion_matrix(y_true2, y_pred2, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    se = tp / (tp + fn + 1e-10)
    sp = tn / (tn + fp + 1e-10)

    return {
        "ICBHI": float(0.5 * (se + sp)),
        "SE": float(se), "SP": float(sp),
        "ACC": float(accuracy_score(y_true2, y_pred2)),
        "F1": float(f1_score(y_true2, y_pred2, zero_division=0)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


# ============================================================
# 3) Train：重点修正了 Loss 计算的类型冲突
# ============================================================
def train_one_epoch(model, loader, device, optim, scheduler, loss_fn, scaler, amp):
    model.train()
    running_loss = 0.0

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        # 规范化：使用最新的 torch.amp 接口
        with torch.amp.autocast('cuda', enabled=amp):
            logits, _ = model(x, mask=mask)

            # 🔥 终极修正：显式将 logits 转为 float32
            # 这样 loss_fn 内部就不会因为半精度权重和长整型标签产生类型推导冲突
            loss = loss_fn(logits.float(), y.long())

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # 梯度剪裁前 unscale
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optim)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ============================================================
# 4) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens")
    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_heavy_12layer")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    train_csv = Path(args.root) / "train_index.csv"
    test_csv = Path(args.root) / "test_index.csv"
    ds_train = TokenNPYDataset(str(train_csv))
    ds_test = TokenNPYDataset(str(test_csv))
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad, num_workers=4,
                          pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=4)

    # Model
    model = build_model(in_dim=args.in_dim, d_model=args.d_model, n_layers=args.n_layers).to(device)

    # Loss：显式指定权重为 float32 且移动到 device
    counts = torch.tensor(ds_train.class_counts, dtype=torch.float32)
    weight = 1.0 / (counts + 1e-6)
    weight = (weight / weight.sum()) * 4.0
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device).float())

    # Optim
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(dl_train))

    # 规范化：使用新的 GradScaler 接口
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best_icbhi = 0.0
    print(f"🚀 开始训练 Heavy SSA 12层模型 | 显存目标: 24GB | BatchSize: {args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, dl_train, device, optim, scheduler, loss_fn, scaler, args.amp)
        metrics = evaluate_icbhi(model, dl_test, device)

        icbhi = metrics["ICBHI"]
        status = "⭐" if icbhi > best_icbhi else " "
        if icbhi > best_icbhi:
            best_icbhi = icbhi
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

        duration = time.time() - t0
        print(f"{status} Epoch {epoch:02d} | Loss: {loss:.4f} | ICBHI: {icbhi:.4f} | "
              f"SE: {metrics['SE']:.4f} | SP: {metrics['SP']:.4f} | F1: {metrics['F1']:.4f} | {duration:.1f}s")

    print(f"✅ 训练完成。最高 ICBHI: {best_icbhi:.4f}")


if __name__ == "__main__":
    main()