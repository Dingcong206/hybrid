#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# ============================================================
# 0) 确保能导入 mymodels.model（用于 2D tokens 模式）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import build_model  # 你自己的序列模型（SSA）


# ✅ 默认 root：请改成你 “ViT 后输出特征” 的目录
DEFAULT_ROOT = "/data/dingcong/hybrid/icbhi_hear_vit_outputs"


# ============================================================
# 1) Dataset：读取你预处理保存的 npy
# ============================================================
class NPYDataset(Dataset):
    def __init__(self, csv_path: str, binary: bool = True):
        self.df = pd.read_csv(csv_path)
        need_cols = ["tokens_path", "label"]
        for c in need_cols:
            if c not in self.df.columns:
                raise ValueError(f"CSV 缺少列: {c}，请检查 {csv_path}")
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npy_path = row["tokens_path"]
        y = int(row["label"])

        # 4-class -> binary: 0=normal, 1/2/3=abnormal
        if self.binary:
            y = 0 if y == 0 else 1

        x = np.load(npy_path)

        # 统一去掉 batch 维（有些保存可能是 (1,T,D) 或 (1,D)）
        if x.ndim >= 1 and x.shape[0] == 1:
            x = np.squeeze(x, axis=0)

        x = torch.from_numpy(x).float()
        return x, torch.tensor(y, dtype=torch.long)


# ============================================================
# 2) Collate：tokens(2D) vs embedding(1D)
# ============================================================
def collate_pad_tokens(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    tokens: x=(T,D)
    return:
      x_pad: (B, T_max, D)
      mask:  (B, T_max) True=PAD
      y:     (B,)
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


def collate_stack_embed(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    embedding: x=(D,)
    return:
      x: (B, D)
      mask: None
      y: (B,)
    """
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys).view(-1)
    return x, None, y


# ============================================================
# 3) Embedding 模式下的 MLP 分类器（不改你 mymodels 也能训练）
# ============================================================
class EmbeddingMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask=None):
        # x: (B, D)
        return self.net(x).squeeze(-1)


# ============================================================
# 4) 指标：ICBHI = (SE + SP)/2
# ============================================================
@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_y = [], []

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        y = y.to(device, non_blocking=True)

        out = model(x, mask)
        file_logit = out[0] if isinstance(out, (tuple, list)) else out

        all_logits.append(file_logit.detach().float().cpu())
        all_y.append(y.detach().long().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()
    return logits, y_true


def compute_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (probs >= thr).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn + 1e-10)
    sp = tn / (tn + fp + 1e-10)
    icbhi = 0.5 * (se + sp)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "ICBHI": float(icbhi),
        "SE": float(se),
        "SP": float(sp),
        "ACC": float(acc),
        "F1": float(f1),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


@torch.no_grad()
def evaluate_with_thr_sweep(model, loader, device, thr_grid=None) -> Dict[str, float]:
    logits, y_true = collect_logits(model, loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))

    if thr_grid is None:
        thr_grid = np.linspace(0.05, 0.95, 19)

    best = None
    for thr in thr_grid:
        m = compute_metrics_from_probs(y_true, probs, float(thr))
        m["thr"] = float(thr)
        if best is None or m["ICBHI"] > best["ICBHI"]:
            best = m
    return best


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 5) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # ✅ root 默认就是 ViT 输出目录，不写也能跑
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT,
                        help="包含 train_index.csv / test_index.csv 的目录（ViT输出特征目录）")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--binary", action="store_true", default=True,
                        help="二分类：normal vs abnormal（默认开启）")

    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_vit",
                        help="保存 best 模型的目录")
    parser.add_argument("--patience", type=int, default=10)

    # tokens 模式下（2D）模型超参
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    # embedding 模式下（1D）MLP 超参
    parser.add_argument("--mlp_hidden", type=int, default=256)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"开始训练: {args.epochs} epochs | device={device}")

    root = Path(args.root)
    train_csv = root / "train_index.csv"
    test_csv = root / "test_index.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"找不到 train/test csv：\n{train_csv}\n{test_csv}\n"
                                f"请确认 --root 指向包含这两个文件的目录。")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model_vit_outputs.pt")

    # ---------- 读 train/test ----------
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # 为 group split 准备 recording（避免泄漏）
    if "recording" not in train_df.columns:
        def infer_recording(p: str) -> str:
            name = Path(p).stem
            return name.split("_cycle")[0] if "_cycle" in name else name
        train_df["recording"] = train_df["tokens_path"].apply(infer_recording)

    # ---------- train -> train_sub / val（按 recording 分组） ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["recording"]))
    train_sub = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_sub_csv = root / "train_sub.csv"
    val_csv = root / "val.csv"
    train_sub.to_csv(train_sub_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"[INFO] train_sub: {len(train_sub)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"[INFO] saved splits: {train_sub_csv} and {val_csv}")

    # ---------- Dataset ----------
    ds_train = NPYDataset(str(train_sub_csv), binary=args.binary)
    ds_val = NPYDataset(str(val_csv), binary=args.binary)
    ds_test = NPYDataset(str(test_csv), binary=args.binary)

    # ✅ 自动判断 feature 形态：tokens(2D) or embedding(1D)
    sample_x, _ = ds_train[0]
    if sample_x.ndim == 2:
        feature_mode = "tokens"
        in_dim = int(sample_x.shape[1])
        collate_fn = collate_pad_tokens
    elif sample_x.ndim == 1:
        feature_mode = "embedding"
        in_dim = int(sample_x.shape[0])
        collate_fn = collate_stack_embed
    else:
        raise ValueError(f"Unsupported sample ndim={sample_x.ndim}, shape={tuple(sample_x.shape)}")

    print(f"[INFO] feature_mode={feature_mode} | sample_shape={tuple(sample_x.shape)} | in_dim={in_dim}")

    # ---------- DataLoader ----------
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # ---------- loss ----------
    if args.binary:
        y_train = train_sub["label"].values
        y_train = np.where(y_train == 0, 0, 1)
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([neg / (pos + 1e-6)], device=device)
        print(f"[INFO] class stats (train_sub): neg={int(neg)} pos={int(pos)} pos_weight={pos_weight.item():.4f}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError("当前脚本默认二分类。")

    # ---------- build model ----------
    if feature_mode == "tokens":
        # 用你自己的 SSA 序列模型
        model = build_model(
            in_dim=in_dim, d_model=args.d_model, n_layers=args.n_layers,
            nhead=args.nhead, dropout=args.dropout
        ).to(device)
    else:
        # embedding 模式：用 MLP（不依赖 mymodels）
        model = EmbeddingMLP(in_dim=in_dim, hidden=args.mlp_hidden, dropout=args.dropout).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ EmbeddingMLP Initialized. Parameters: {params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(dl_train))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    best_icbhi, best_thr, best_epoch = -1.0, 0.5, -1
    bad_epochs = 0

    # ---------- training loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for x, mask, y in dl_train:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True) if mask is not None else None
            y = y.to(device, non_blocking=True).float()

            out = model(x, mask)
            file_logit = out[0] if isinstance(out, (tuple, list)) else out

            loss = loss_fn(file_logit.view(-1), y.view(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            scheduler.step()

            running += float(loss.item())

        train_loss = running / max(1, len(dl_train))

        val_best = evaluate_with_thr_sweep(model, dl_val, device)
        icbhi, se, sp, thr = val_best["ICBHI"], val_best["SE"], val_best["SP"], val_best["thr"]

        improved = icbhi > best_icbhi + 1e-6
        if improved:
            best_icbhi, best_thr, best_epoch = icbhi, thr, epoch
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_icbhi": best_icbhi,
                "best_thr": best_thr,
                "args": vars(args),
                "feature_mode": feature_mode,
                "in_dim": in_dim,
            }, ckpt_path)
            star = "⭐"
        else:
            bad_epochs += 1
            star = " "

        dt = time.time() - t0
        print(
            f"{star} Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | "
            f"VAL ICBHI {icbhi:.4f} (best {best_icbhi:.4f} @ep{best_epoch}) | "
            f"thr {thr:.2f} | SE {se:.4f} SP {sp:.4f} | "
            f"ACC {val_best['ACC']:.4f} F1 {val_best['F1']:.4f} | "
            f"TP {val_best['TP']} TN {val_best['TN']} FP {val_best['FP']} FN {val_best['FN']} | "
            f"{dt:.1f}s"
        )

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] val ICBHI {args.patience} 轮无提升，停止于 epoch {epoch}.")
            break

    # ---------- final test ----------
    print("\n[FINAL] Loading best checkpoint and evaluating on TEST once...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    best_thr = float(ckpt.get("best_thr", best_thr))

    logits, y_true = collect_logits(model, dl_test, device)
    probs = 1.0 / (1.0 + np.exp(-logits))
    test_metrics = compute_metrics_from_probs(y_true, probs, best_thr)

    print(
        f"[TEST] thr={best_thr:.2f} | "
        f"ICBHI {test_metrics['ICBHI']:.4f} SE {test_metrics['SE']:.4f} SP {test_metrics['SP']:.4f} | "
        f"ACC {test_metrics['ACC']:.4f} F1 {test_metrics['F1']:.4f} | "
        f"TP {test_metrics['TP']} TN {test_metrics['TN']} FP {test_metrics['FP']} FN {test_metrics['FN']}"
    )
    print(f"[SAVED] best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
