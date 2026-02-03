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
# 0) 让 `from mymodels.model import build_model` 能导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mymodels.model import build_model   # ✅ 方案B：model 内部直接输出 (B,4)


# ============================================================
# 1) Dataset：读取预处理保存的 tokens.npy（四分类 label 0/1/2/3）
# ============================================================
class TokenNPYDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        for c in ["tokens_path", "label"]:
            if c not in self.df.columns:
                raise ValueError(f"CSV 缺少列: {c}，请检查 {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npy_path = row["tokens_path"]
        y = int(row["label"])               # 0/1/2/3

        x = np.load(npy_path)               # (T, D)
        x = torch.from_numpy(x).float()
        return x, torch.tensor(y, dtype=torch.long)


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
# 2) 评估：patch-mix 风格 argmax + 4->2 ICBHI
# ============================================================
@torch.no_grad()
def evaluate_like_patchmix(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_pred4, all_true4 = [], []

    for x, mask, y in loader:
        x = x.to(device)
        mask = mask.to(device)
        y = y.to(device)

        out = model(x, mask)
        file_logits = out[0] if isinstance(out, (tuple, list)) else out  # ✅ (B,4)

        pred4 = torch.argmax(file_logits, dim=1)  # 4-class argmax

        all_pred4.append(pred4.cpu())
        all_true4.append(y.cpu())

    y_pred4 = torch.cat(all_pred4).numpy()
    y_true4 = torch.cat(all_true4).numpy()

    # 4->2: 0=normal, 1/2/3=abnormal
    y_pred2 = (y_pred4 != 0).astype(np.int64)
    y_true2 = (y_true4 != 0).astype(np.int64)

    cm = confusion_matrix(y_true2, y_pred2, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn + 1e-10)
    sp = tn / (tn + fp + 1e-10)
    icbhi = 0.5 * (se + sp)

    acc = accuracy_score(y_true2, y_pred2)
    f1 = f1_score(y_true2, y_pred2, zero_division=0)

    return {
        "ICBHI": float(icbhi),
        "SE": float(se),
        "SP": float(sp),
        "ACC": float(acc),
        "F1": float(f1),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


# ============================================================
# 3) 工具：随机种子
# ============================================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 4) 主训练流程：patch-mix 风格：每 epoch 在 TEST 上评估并选 best
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",
        help="预处理输出目录（包含 train_index.csv / test_index.csv）"
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_patchmix_style")
    parser.add_argument("--patience", type=int, default=10)

    # tokens 维度（来自 AST patch projection hidden_dim，常见 768）
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_classes", type=int, default=4)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    root = Path(args.root)
    train_csv = root / "train_index.csv"
    test_csv = root / "test_index.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"找不到：\n{train_csv}\n{test_csv}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pt")

    # Dataset / Loader
    ds_train = TokenNPYDataset(str(train_csv))
    ds_test = TokenNPYDataset(str(test_csv))

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate_pad)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True, collate_fn=collate_pad)

    print(f"[INFO] train cycles: {len(ds_train)} | test cycles: {len(ds_test)}")

    # ✅ 方案B：直接 build 4-class 模型
    model = build_model(
        in_dim=args.in_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        num_classes=args.num_classes
    ).to(device)

    # ✅ sanity check：确认 file_logits 是 (B,4)
    x0, m0, y0 = next(iter(dl_train))
    with torch.no_grad():
        out0 = model(x0.to(device), m0.to(device))
        file_logits0 = out0[0] if isinstance(out0, (tuple, list)) else out0
    print("[DEBUG] file_logits shape (must be B,4):", tuple(file_logits0.shape))

    # loss/optim/scheduler
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(dl_train))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    best_icbhi = -1.0
    best_epoch = -1
    bad_epochs = 0

    print("\n🚀 Start training (patch-mix style: eval on TEST every epoch)\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for x, mask, y in dl_train:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)  # long, 0~3

            out = model(x, mask)
            file_logits = out[0] if isinstance(out, (tuple, list)) else out  # ✅ (B,4)

            loss = loss_fn(file_logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            scheduler.step()

            running += float(loss.item())

        train_loss = running / max(1, len(dl_train))

        # ✅ 每轮评估 TEST，按 ICBHI 选 best（像 patch-mix）
        test_m = evaluate_like_patchmix(model, dl_test, device)
        icbhi = test_m["ICBHI"]

        improved = icbhi > best_icbhi + 1e-6
        if improved:
            best_icbhi = icbhi
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "best_icbhi": best_icbhi, "args": vars(args)},
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
            f"TEST ICBHI {test_m['ICBHI']:.4f} SE {test_m['SE']:.4f} SP {test_m['SP']:.4f} | "
            f"ACC {test_m['ACC']:.4f} F1 {test_m['F1']:.4f} | "
            f"TP {test_m['TP']} TN {test_m['TN']} FP {test_m['FP']} FN {test_m['FN']} | "
            f"{dt:.1f}s"
        )

        if bad_epochs >= args.patience:
            print(f"[EARLY STOP] TEST ICBHI 连续 {args.patience} 轮无提升，停止于 epoch {epoch}（best@{best_epoch}）")
            break

    print(f"\n✅ DONE. Best ICBHI={best_icbhi:.4f} @ epoch {best_epoch}")
    print(f"[SAVED] best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
