#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from mamba_ssm import Mamba


# =========================================================
# 0) Repro
# =========================================================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1) Your model (no downsample) for input (B, 948, 768)
# =========================================================
def sinusoidal_positional_encoding(seq_len: int, dim: int, device):
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h = self.fwd(h) + torch.flip(self.bwd(torch.flip(h, [1])), [1])
        x = x + self.drop(h)
        return x + self.mlp(self.ln2(x))


class SSA_Layer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.mamba = BiMambaBlock(d_model, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x, mask=None):
        res = x
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.mamba(x)

        x_n = self.attn_ln(x)
        x_a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=mask, need_weights=False)
        x = x + x_a

        g = self.gate(x.mean(dim=1, keepdim=True))
        return res + g * x


class SSA_Model_NoDownsample(nn.Module):
    """
    输入：AST patch tokens (B, 948, 768)
    输出：file_logit (B,), token_logits (B, 948) [可选监控]
    """
    def __init__(self, in_dim=768, d_model=256, n_layers=4, nhead=8, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList([
            SSA_Layer(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        self.classifier = nn.Linear(d_model, 1)
        self.patch_head = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: (B, 948, 768)
        x = self.input_proj(x)  # (B, 948, 256)

        B, T, D = x.shape
        pos = sinusoidal_positional_encoding(T, D, x.device).unsqueeze(0)
        x = x + pos

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)  # (B, 948, 256)

        attn_weights = self.attention_net(x)  # (B, 948, 1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)

        file_feature = torch.sum(attn_weights * x, dim=1)          # (B, 256)
        file_logit = self.classifier(file_feature).squeeze(-1)     # (B,)
        token_logits = self.patch_head(x).squeeze(-1)              # (B, 948)
        return file_logit, token_logits


def build_model(in_dim=768, d_model=256, n_layers=4, nhead=8, dropout=0.3):
    model = SSA_Model_NoDownsample(
        in_dim=in_dim, d_model=d_model, n_layers=n_layers, nhead=nhead, dropout=dropout
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model initialized. Params: {params:,}")
    return model


# =========================================================
# 2) Dataset: read train_index.csv / test_index.csv
# =========================================================
class TokenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, binary: bool = True):
        """
        df must contain: tokens_path, label
        binary=True: y = 0(normal) vs 1(abnormal=label!=0)
        """
        self.df = df.reset_index(drop=True)
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["tokens_path"]
        x = np.load(path).astype(np.float32)  # (948, 768)
        y4 = int(row["label"])
        if self.binary:
            y = 0 if y4 == 0 else 1
        else:
            y = y4
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def stratified_split(df: pd.DataFrame, val_ratio=0.1, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by label column.
    """
    rng = np.random.default_rng(seed)
    labels = df["label"].values
    train_idx, val_idx = [], []

    for lab in np.unique(labels):
        idxs = np.where(labels == lab)[0]
        rng.shuffle(idxs)
        n_val = int(math.ceil(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# =========================================================
# 3) Metrics: ICBHI score (binary Normal vs Abnormal)
# =========================================================
@torch.no_grad()
def compute_metrics_binary(logits: np.ndarray, y_true: np.ndarray, thr: float = 0.5):
    """
    logits: raw logits
    y_true: 0/1
    """
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= thr).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    se = tp / (tp + fn + 1e-10)  # sensitivity (abnormal)
    sp = tn / (tn + fp + 1e-10)  # specificity (normal)
    icbhi = (se + sp) / 2.0

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    # F1
    prec = tp / (tp + fp + 1e-10)
    rec = se
    f1 = 2 * prec * rec / (prec + rec + 1e-10)

    return {
        "ICBHI": float(icbhi),
        "SE": float(se),
        "SP": float(sp),
        "ACC": float(acc),
        "F1": float(f1),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


# =========================================================
# 4) Train / Eval
# =========================================================
def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)  # (B, 948, 768)
        y = y.to(device)  # (B,)
        optimizer.zero_grad(set_to_none=True)
        file_logit, _ = model(x)
        loss = loss_fn(file_logit, y.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device)
        file_logit, _ = model(x)
        all_logits.append(file_logit.detach().cpu().numpy())
        all_y.append(y.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(np.int64)
    return compute_metrics_binary(logits, y_true, thr=thr)


# =========================================================
# 5) Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",
                        help="包含 train_index.csv / test_index.csv 的目录")
    parser.add_argument("--binary", action="store_true", help="二分类：normal vs abnormal（推荐 ICBHI 指标）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="从 official train 中划出 val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_sampler", action="store_true", help="用 WeightedRandomSampler 缓解不均衡")
    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/checkpoints_ssa_mamba")
    args = parser.parse_args()

    seed_all(args.seed)

    train_csv = os.path.join(args.root, "train_index.csv")
    test_csv = os.path.join(args.root, "test_index.csv")
    assert os.path.exists(train_csv), f"找不到 {train_csv}"
    assert os.path.exists(test_csv), f"找不到 {test_csv}"

    df_train_all = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # official train -> split train/val
    df_train, df_val = stratified_split(df_train_all, val_ratio=args.val_ratio, seed=args.seed)

    print(f"[INFO] train cycles: {len(df_train)}  val cycles: {len(df_val)}  test cycles: {len(df_test)}")
    print("[INFO] train label counts:\n", df_train["label"].value_counts().sort_index())
    print("[INFO] val label counts:\n", df_val["label"].value_counts().sort_index())
    print("[INFO] test label counts:\n", df_test["label"].value_counts().sort_index())

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] cuda 不可用，切到 cpu")
    else:
        device = torch.device(args.device)

    # datasets
    binary = True if args.binary else True  # 默认按 ICBHI 二分类跑
    train_ds = TokenDataset(df_train, binary=binary)
    val_ds = TokenDataset(df_val, binary=binary)
    test_ds = TokenDataset(df_test, binary=binary)

    # sampler (optional)
    if args.use_sampler:
        # class weights from binary labels
        y_train = (df_train["label"].values != 0).astype(np.int64)
        class_counts = np.bincount(y_train, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
        print(f"[INFO] sampler enabled. binary counts={class_counts.tolist()}, weights={class_weights.tolist()}")
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # model
    model = build_model(in_dim=768, d_model=256, n_layers=4, nhead=8, dropout=0.3).to(device)

    # loss: binary BCEWithLogits
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    best_icbhi = -1.0
    best_path = os.path.join(args.save_dir, "best_icbhi.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics = evaluate(model, val_loader, device, thr=0.5)

        print(
            f"Epoch {epoch:02d} | loss {tr_loss:.4f} | "
            f"VAL ICBHI {val_metrics['ICBHI']:.4f} SE {val_metrics['SE']:.4f} SP {val_metrics['SP']:.4f} "
            f"ACC {val_metrics['ACC']:.4f} F1 {val_metrics['F1']:.4f} "
            f"TP {val_metrics['TP']} TN {val_metrics['TN']} FP {val_metrics['FP']} FN {val_metrics['FN']}"
        )

        if val_metrics["ICBHI"] > best_icbhi:
            best_icbhi = val_metrics["ICBHI"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, best_path)
            print(f"⭐ Saved best model: {best_path} (ICBHI={best_icbhi:.4f})")

    # final test with best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, thr=0.5)

    print("\n===== ✅ FINAL TEST (best by VAL ICBHI) =====")
    print(
        f"TEST ICBHI {test_metrics['ICBHI']:.4f} SE {test_metrics['SE']:.4f} SP {test_metrics['SP']:.4f} "
        f"ACC {test_metrics['ACC']:.4f} F1 {test_metrics['F1']:.4f} "
        f"TP {test_metrics['TP']} TN {test_metrics['TN']} FP {test_metrics['FP']} FN {test_metrics['FN']}"
    )


if __name__ == "__main__":
    main()
