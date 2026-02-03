#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Make project imports work
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import build_model  # noqa


# =========================
# Repro
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class TokenDataset(Dataset):
    """
    Reads train_index.csv / test_index.csv
    Required columns:
      - tokens_path
      - label (0/1/2/3)
    """
    def __init__(self, csv_path: str, binary: bool = True):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.binary = binary
        for col in ["tokens_path", "label"]:
            if col not in self.df.columns:
                raise ValueError(f"[ERR] CSV缺少列 {col}: {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"]).astype(np.float32)  # (948, 768)
        y4 = int(row["label"])

        if self.binary:
            y = 0 if y4 == 0 else 1
            y = torch.tensor(y, dtype=torch.float32)  # BCE需要float
        else:
            y = torch.tensor(y4, dtype=torch.long)

        return torch.from_numpy(x), y


# =========================
# Metrics (binary ICBHI)
# =========================
@torch.no_grad()
def evaluate_binary(model, loader, device, thr=0.5):
    model.eval()
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device)              # (B,948,768)
        y = y.to(device)              # (B,)
        out = model(x)

        # 兼容两种返回：file_logit 或 (file_logit, logits)
        if isinstance(out, (tuple, list)):
            file_logit = out[0]
        else:
            file_logit = out

        all_logits.append(file_logit.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0).reshape(-1)
    y_true = np.concatenate(all_y, axis=0).astype(np.int64)

    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= thr).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    se = tp / (tp + fn + 1e-10)
    sp = tn / (tn + fp + 1e-10)
    icbhi = (se + sp) / 2.0

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    prec = tp / (tp + fp + 1e-10)
    f1 = 2 * prec * se / (prec + se + 1e-10)

    return {
        "ICBHI": float(icbhi),
        "SE": float(se),
        "SP": float(sp),
        "ACC": float(acc),
        "F1": float(f1),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


# =========================
# Train
# =========================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        file_logit = out[0] if isinstance(out, (tuple, list)) else out

        loss = loss_fn(file_logit.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str,
                        default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",
                        help="包含 train_index.csv / test_index.csv 的目录")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thr", type=float, default=0.5)

    # 你的 tokens 是 (948,768)
    parser.add_argument("--in_dim", type=int, default=768, help="token feature dim")
    parser.add_argument("--d_model", type=int, default=256, help="your SSA internal dim")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)

    # 严谨：不拿test做早停；但你如果想保存最后模型可以开
    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/ckpt_strict")
    parser.add_argument("--save_last", action="store_true")

    args = parser.parse_args()
    seed_all(args.seed)

    train_csv = os.path.join(args.root, "train_index.csv")
    test_csv = os.path.join(args.root, "test_index.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"[ERR] not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"[ERR] not found: {test_csv}")

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] CUDA不可用，切到CPU")
    else:
        device = torch.device(args.device)

    train_ds = TokenDataset(train_csv, binary=True)
    test_ds = TokenDataset(test_csv, binary=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"[INFO] OFFICIAL TRAIN cycles: {len(train_ds)}")
    print(f"[INFO] OFFICIAL TEST  cycles: {len(test_ds)}")

    # ✅ build your model
    model = build_model(in_dim=args.in_dim, d_model=args.d_model, n_layers=args.n_layers, nhead=args.nhead)
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)

    # ============ training ============
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # 这里我给你“作者风格”的日志：每个epoch看一下train loss +（可选）test指标
        # 如果你想“最严谨”，你可以注释掉下面这段，让它只在最后评一次 test
        metrics = evaluate_binary(model, test_loader, device, thr=args.thr)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {tr_loss:.4f} | "
            f"TEST ICBHI {metrics['ICBHI']:.4f} "
            f"SE {metrics['SE']:.4f} SP {metrics['SP']:.4f} "
            f"ACC {metrics['ACC']:.4f} F1 {metrics['F1']:.4f} "
            f"TP {metrics['TP']} TN {metrics['TN']} FP {metrics['FP']} FN {metrics['FN']}"
        )

    # ============ final one-shot test ============
    final_metrics = evaluate_binary(model, test_loader, device, thr=args.thr)
    print("\n===== ✅ FINAL TEST (OFFICIAL 60/40) =====")
    print(
        f"ICBHI {final_metrics['ICBHI']:.4f} | "
        f"SE {final_metrics['SE']:.4f} | SP {final_metrics['SP']:.4f} | "
        f"ACC {final_metrics['ACC']:.4f} | F1 {final_metrics['F1']:.4f} | "
        f"TP {final_metrics['TP']} TN {final_metrics['TN']} FP {final_metrics['FP']} FN {final_metrics['FN']}"
    )

    if args.save_last:
        ckpt_path = os.path.join(args.save_dir, "last_epoch.pt")
        torch.save(
            {"model": model.state_dict(),
             "final_metrics": final_metrics,
             "args": vars(args)},
            ckpt_path
        )
        print(f"[INFO] saved: {ckpt_path}")


if __name__ == "__main__":
    main()