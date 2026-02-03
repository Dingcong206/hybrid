# !/usr/bin/env python3
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
# 路径兼容处理
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入你的模型
try:
    from mymodels.model import build_model
except ImportError:
    print("[ERR] 请确保 mymodels/model.py 路径正确且包含 build_model 函数")
    sys.exit(1)


# =========================
# 随机种子设置
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset (适配 tokens)
# =========================
class TokenDataset(Dataset):
    def __init__(self, csv_path: str, binary: bool = True):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"]).astype(np.float32)  # (Seq_len, 768)
        y4 = int(row["label"])

        if self.binary:
            # ICBHI 标准二分类逻辑
            y = 0 if y4 == 0 else 1
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = torch.tensor(y4, dtype=torch.long)
        return torch.from_numpy(x), y


# =========================
# 指标评估 (完全对齐作者公式)
# =========================
@torch.no_grad()
def evaluate_binary(model, loader, device, thr=0.5):
    model.eval()
    all_logits = []
    all_y = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        # 兼容 (logit, feature) 输出格式
        file_logit = out[0] if isinstance(out, (tuple, list)) else out
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
    acc = (tp + tn) / (len(y_true) + 1e-10)

    return {
        "ICBHI": float(icbhi), "SE": float(se), "SP": float(sp), "ACC": float(acc),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }


# =========================
# 单轮训练
# =========================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        file_logit = out[0] if isinstance(out, (tuple, list)) else out

        loss = loss_fn(file_logit.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
    return total_loss / len(loader.dataset)


# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/dingcong/hybrid/icbhi_official_ast_patch_tokens")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_author_style")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thr", type=float, default=0.5)
    args = parser.parse_args()

    seed_all(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    train_loader = DataLoader(TokenDataset(os.path.join(args.root, "train_index.csv")),
                              batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(TokenDataset(os.path.join(args.root, "test_index.csv")),
                             batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = build_model(in_dim=768).to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- 作者风格的关键变量 ---
    best_icbhi = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(args.save_dir, "best_model.pt")

    print(f"开始训练: {args.epochs} Epochs, 使用设备: {args.device}")

    for epoch in range(1, args.epochs + 1):
        # 1. 训练
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, args.device)

        # 2. 评估 (像作者一样每个 Epoch 都测)
        metrics = evaluate_binary(model, test_loader, args.device, thr=args.thr)
        curr_icbhi = metrics["ICBHI"]

        # 3. 追踪并保存最佳模型 (Core Logic)
        # 增加 SE > 0.05 是为了确保模型不是通过把所有样本猜成一类来骗分的
        if curr_icbhi > best_icbhi and metrics["SE"] > 0.05:
            best_icbhi = curr_icbhi
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
            print(f" ⭐ [Epoch {epoch}] 发现更优模型! ICBHI: {curr_icbhi:.4f}")

        # 打印日志
        print(f"Epoch [{epoch:03d}/{args.epochs}] Loss: {tr_loss:.4f} | "
              f"Score: {curr_icbhi:.4f} (Best: {best_icbhi:.4f}) | "
              f"SE: {metrics['SE']:.4f} SP: {metrics['SP']:.4f}")

    # ==================================
    # 最终步骤：加载历史上表现最好的一版进行测试汇报
    # ==================================
    print("\n" + "=" * 40)
    print(f"训练完成！正在加载第 {best_epoch} 轮的最佳权重进行最终评估...")

    best_ckpt = torch.load(checkpoint_path)
    model.load_state_dict(best_ckpt['model_state_dict'])

    final_metrics = evaluate_binary(model, test_loader, args.device, thr=args.thr)

    print("\n===== ✅ FINAL TEST RESULTS (Author Style) =====")
    print(f"Best Epoch: {best_epoch}")
    print(f"ICBHI Score: {final_metrics['ICBHI']:.4f}")
    print(f"Sensitivity (SE): {final_metrics['SE']:.4f}")
    print(f"Specificity (SP): {final_metrics['SP']:.4f}")
    print(f"Accuracy: {final_metrics['ACC']:.4f}")
    print(
        f"Confusion: TP={final_metrics['TP']}, TN={final_metrics['TN']}, FP={final_metrics['FP']}, FN={final_metrics['FN']}")
    print("=" * 40)


if __name__ == "__main__":
    main()