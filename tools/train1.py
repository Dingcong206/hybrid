#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import build_model
from util.misc import AverageMeter, accuracy, warmup_learning_rate, adjust_learning_rate


# ============================================================
# 1) Dataset & Loader (保持你的 Token 加载逻辑)
# ============================================================
class TokenNPYDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.labels = self.df["label"].astype(int).values
        self.class_counts = np.bincount(self.labels, minlength=4)

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


def set_loader(args):
    train_csv = Path(args.root) / "train_index.csv"
    test_csv = Path(args.root) / "test_index.csv"

    train_dataset = TokenNPYDataset(str(train_csv))
    val_dataset = TokenNPYDataset(str(test_csv))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pad, num_workers=args.num_workers)

    args.class_counts = train_dataset.class_counts
    return train_loader, val_loader, args


# ============================================================
# 2) Model & Criterion
# ============================================================
def set_model(args, device):
    model = build_model(in_dim=args.in_dim, d_model=args.d_model, n_layers=args.n_layers).to(device)

    # 权重 Loss 计算
    counts = torch.tensor(args.class_counts, dtype=torch.float32)
    weight = 1.0 / (counts + 1e-6)
    weight = (weight / weight.sum()) * 4.0
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return model, criterion, optimizer


# ============================================================
# 3) Core Functions (Train & Validate)
# ============================================================
def train(train_loader, model, criterion, optimizer, epoch, args, scaler, device):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, mask, labels) in enumerate(train_loader):
        images, mask, labels = images.to(device), mask.to(device), labels.to(device)
        bsz = labels.shape[0]

        # Warmup
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.amp.autocast('cuda', enabled=args.amp):
            logits, _ = model(images, mask=mask)
            loss = criterion(logits.float(), labels)

        losses.update(loss.item(), bsz)
        acc1, _ = accuracy(logits, labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print(f'Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args, device):
    model.eval()
    all_pred2, all_true2 = [], []

    with torch.no_grad():
        for x, mask, y in val_loader:
            x, mask = x.to(device), mask.to(device)
            logits, _ = model(x, mask=mask)

            # ICBHI 2-class 逻辑: 0是正常，1,2,3是非正常
            pred4 = torch.argmax(logits, dim=1).cpu()
            all_pred2.append((pred4 != 0).long())
            all_true2.append((y != 0).long())

    y_pred2 = torch.cat(all_pred2).numpy()
    y_true2 = torch.cat(all_true2).numpy()

    # 手动计算 ICBHI Score (Sp, Se)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true2, y_pred2, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn + 1e-10) * 100
    sp = tn / (tn + fp + 1e-10) * 100
    sc = (se + sp) / 2

    return sp, se, sc


# ============================================================
# 4) Main Pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser('ICBHI Token Training')
    # 基本设定
    parser.add_argument('--root', type=str, default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens")
    parser.add_argument('--save_dir', type=str, default="/data/dingcong/hybrid/checkpoints_ssamamba")
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    # 优化设定
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warm', action='store_true', default=True)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    # 模型设定
    parser.add_argument('--in_dim', type=int, default=768)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    # 1. 加载数据
    train_loader, val_loader, args = set_loader(args)

    # 2. 构建模型
    model, criterion, optimizer = set_model(args, device)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    best_acc = [0, 0, 0]  # Sp, Se, Score
    os.makedirs(args.save_dir, exist_ok=True)

    # 3. 循环训练
    for epoch in range(1, args.epochs + 1):
        # 模拟参考代码的 LR 调整逻辑 (可以根据需要换成 Cosine)
        adjust_learning_rate(args, optimizer, epoch)

        t1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args, scaler, device)
        sp, se, sc = validate(val_loader, model, criterion, args, device)

        # 保存最优模型 (依据 ICBHI Score)
        if sc > best_acc[2]:
            best_acc = [sp, se, sc]
            save_file = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({'model': model.state_dict(), 'args': args, 'epoch': epoch}, save_file)
            print(f'==> Best Score Updated: {sc:.2f}')

        print(f'Epoch {epoch} Time {time.time() - t1:.1f}s | Loss {loss:.4f} | '
              f'Sp: {sp:.2f} Se: {se:.2f} Score: {sc:.2f} (Best Score: {best_acc[2]:.2f})')

    # 保存最终结果
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({"best_sp": best_acc[0], "best_se": best_acc[1], "best_score": best_acc[2]}, f)


if __name__ == '__main__':
    main()