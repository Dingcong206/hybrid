
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mymodels.model import build_model  # ✅ 直接用你的模型


# =========================
# 0) Repro
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1) Dataset: tokens_path + label -> binary label (ICBHI)
# =========================
class TokenDataset(Dataset):
    """
    读取你生成的 tokens（AST patch projection 输出）
    CSV需要至少包含：
      - tokens_path: .npy路径
      - label: 0/1/2/3 (ICBHI四类)
    严格 ICBHI score 常用：二分类 normal(0) vs abnormal(1/2/3)
    """
    def __init__(self, csv_path: str, binary: bool = True):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.binary = binary

        # 只做最基本检查，避免你跑到一半才报错
        for col in ["tokens_path", "label"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV缺少列 {col}：{csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"]).astype(np.float32)  # (948, 768)
        y4 = int(row["label"])

        if self.binary:
            y = 0 if y4 == 0 else 1
            y = torch.tensor(y, dtype=torch.float32)  # BCEWithLogitsLoss 用 float
        else:
            y = torch.tensor(y4, dtype=torch.long)     # 4类用 CE

        return torch.from_numpy(x), y


# =========================
# 2) Metrics: ICBHI score (binary)
# =========================
@torch.no_grad()
def eval_icbhi_binary(model, loader, device, thr=0.5):
    model.eval()
    all_logits, all_y = [], []

    for x, y in loader:
        x = x.to(device)  # (B, 948, 768)
        y = y.to(device)  # (B,)
        file_logit, _ = model(x)  # ✅ 你的模型返回 (file_logit, logits)
        all_logits.append(file_logit.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(np.int64)

    probs = 1 / (1 + np.exp(-logits))
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
# 3) Train
# =========================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        file_logit, _ = model(x)
        loss = loss_fn(file_logit, y)
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * x.size(0)
    return total / len(loader.dataset)


# =========================
# 4) Main (STRICT OFFICIAL 60/40)
# =========================
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
    parser.add_argument("--save_dir", type=str, default="/data/dingcong/hybrid/ckpt_strict")
    parser.add_argument("--save_last", action="store_true", help="是否保存最后一个epoch模型")
    args = parser.parse_args()

    seed_all(args.seed)

    train_csv = os.path.join(args.root, "train_index.csv")
    test_csv  = os.path.join(args.root, "test_index.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"找不到 {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"找不到 {test_csv}")

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[WARN] CUDA不可用，已切换到 CPU")
    else:
        device = torch.device(args.device)

    # data
    train_ds = TokenDataset(train_csv, binary=True)
    test_ds  = TokenDataset(test_csv,  binary=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[INFO] OFFICIAL TRAIN cycles: {len(train_ds)}")
    print(f"[INFO] OFFICIAL TEST  cycles: {len(test_ds)}")

    # model (✅ 直接用你的 build_model)
    # 你自己的 build_model(in_dim=?, ...) 如果需要参数，你在这里传
    model = build_model().to(device)

    # loss/optim
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)

    # ========= STRICT protocol =========
    # 不切 val；不用 test 选最优；最后只报告一次 test
    # 你可以训练中打印 test 作为观察，但严格论文复现最好只最后评一次
    # 这里默认：每个epoch只打印train loss；最后评一次test
    # ===================================
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {tr_loss:.4f}")

    # final one-shot test
    metrics = eval_icbhi_binary(model, test_loader, device, thr=args.thr)
    print("\n===== ✅ FINAL TEST (OFFICIAL 60/40, STRICT) =====")
    print(
        f"ICBHI {metrics['ICBHI']:.4f} | SE {metrics['SE']:.4f} | SP {metrics['SP']:.4f} | "
        f"ACC {metrics['ACC']:.4f} | F1 {metrics['F1']:.4f} | "
        f"TP {metrics['TP']} TN {metrics['TN']} FP {metrics['FP']} FN {metrics['FN']}"
    )

    if args.save_last:
        ckpt_path = os.path.join(args.save_dir, "last_epoch.pt")
        torch.save(
            {"model": model.state_dict(), "metrics": metrics, "epochs": args.epochs, "seed": args.seed},
            ckpt_path
        )
        print(f"[INFO] saved: {ckpt_path}")


if __name__ == "__main__":
    main()
