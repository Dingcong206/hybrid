import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

from mymodels.model import build_model  # 你的模型：输入(B,T,200,48)->输出(B,T) logits


# =========================
# 1) 配置（与你 baseline 对齐）
# =========================
FEAT_DIR = "/data/dingcong/hybrid/hear_patch_final"          # 每个文件: (T,200,48)
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"     # ICBHI txt

RANDOM_SEED = 42
TEST_SIZE = 0.2

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
WEIGHT_POS = 3.0            # 对齐你的 BEST_WEIGHT（正类权重）
TOP_K = 5                   # 对齐你的 TOP_K
BEST_THRESHOLD = 0.80       # 对齐你的 BEST_THRESHOLD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2) 标签读取（与你 baseline 一样）
# =========================
def get_label(base_name: str):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path):
        return None
    df = pd.read_csv(txt_path, sep="\t", header=None)
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0


# =========================
# 3) Dataset：返回“一个文件”的所有 segments
#    x: (T,200,48)
#    y_file: 0/1
# =========================
class FilePatchDataset(Dataset):
    def __init__(self, file_paths):
        self.items = []
        for f in file_paths:
            base = os.path.basename(f).replace(".npy", "")
            y = get_label(base)
            if y is None:
                continue
            self.items.append((f, int(y), base))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f, y, base = self.items[idx]
        x = np.load(f).astype(np.float32)     # (T,200,48)
        x = torch.from_numpy(x)               # torch float32
        y = torch.tensor(y, dtype=torch.float32)
        return x, y, base


# =========================
# 4) Collate：把不同长度 T 的文件 padding 成同一长度
#    返回：
#      x_pad: (B, Tmax, 200, 48)
#      mask : (B, Tmax)  True=有效token
# =========================
def collate_pad(batch):
    xs, ys, bases = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    Tmax = max(lengths)

    B = len(xs)
    x_pad = torch.zeros((B, Tmax, xs[0].shape[1], xs[0].shape[2]), dtype=torch.float32)
    mask = torch.zeros((B, Tmax), dtype=torch.bool)

    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t] = x
        mask[i, :t] = True

    y = torch.stack(list(ys), dim=0)  # (B,)
    return x_pad, mask, y, list(bases), lengths


# =========================
# 5) Top-K mean（按文件聚合）
# =========================
def topk_mean_1d(probs_1d: torch.Tensor, k: int):
    # probs_1d: (T,)
    kk = min(k, probs_1d.numel())
    topk, _ = torch.topk(probs_1d, kk)
    return topk.mean()


# =========================
# 6) 训练：segment-level BCE（完全等价 baseline）
#    - logits: (B,T)
#    - 将每个 segment 的 label 都设为 file label
# =========================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_pad, mask, y_file, _, _ in loader:
        x_pad = x_pad.to(DEVICE)
        mask = mask.to(DEVICE)
        y_file = y_file.to(DEVICE)
        attn_mask= ~mask
        logits = model(x_pad, patch_mask=attn_mask)  # 期望输出 (B,T)，mask可选（你模型不收也行）
        if logits.dim() != 2:
            raise RuntimeError(f"Model must return (B,T) logits, got {tuple(logits.shape)}")

        # 构造 segment-level label: (B,T) = file label broadcast
        y_seg = y_file.unsqueeze(1).expand_as(logits)

        # 只在有效 segment 上计算 loss
        logits_valid = logits[mask]      # (N_valid,)
        y_valid = y_seg[mask]            # (N_valid,)

        # class imbalance：pos_weight
        pos_weight = torch.tensor([WEIGHT_POS], device=DEVICE)
        loss = F.binary_cross_entropy_with_logits(logits_valid, y_valid, pos_weight=pos_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# =========================
# 7) 测试：file-level Top-K mean + 阈值
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    y_true_file = []
    y_prob_file = []
    y_pred_file = []

    for x_pad, mask, y_file, bases, lengths in loader:
        x_pad = x_pad.to(DEVICE)
        mask = mask.to(DEVICE)
        attn_mask = ~mask
        logits = model(x_pad, patch_mask=attn_mask)      # (B,T)
        probs = torch.sigmoid(logits)         # (B,T)

        # file-level 聚合：对每个文件只取有效长度的 probs，再 top-k mean
        for i in range(probs.shape[0]):
            t = lengths[i]
            p_i = probs[i, :t]  # (t,)
            file_prob = topk_mean_1d(p_i, TOP_K).item()

            y_prob_file.append(file_prob)
            y_true_file.append(int(y_file[i].item()))
            y_pred_file.append(1 if file_prob >= BEST_THRESHOLD else 0)

    cm = confusion_matrix(y_true_file, y_pred_file, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    se = tp / (tp + fn + 1e-9)
    sp = tn / (tn + fp + 1e-9)
    icbhi = 0.5 * (se + sp)

    auc = roc_auc_score(y_true_file, y_prob_file) if len(set(y_true_file)) > 1 else float("nan")

    return {
        "cm": cm,
        "se": se,
        "sp": sp,
        "icbhi": icbhi,
        "auc": auc,
        "y_true": y_true_file,
        "y_pred": y_pred_file,
        "y_prob": y_prob_file,
    }


# =========================
# 8) 主流程
# =========================
def main():
    feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
    train_files, test_files = train_test_split(
        feat_files, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    train_ds = FilePatchDataset(train_files)
    test_ds = FilePatchDataset(test_files)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_pad
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_pad
    )

    model = build_model(in_dim=48).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_icbhi = -1.0
    save_path = "best_deep_baseline_equivalent.pth"

    # sanity check
    x0, m0, y0, _, _ = next(iter(train_loader))
    x0 = x0.to(DEVICE); m0 = m0.to(DEVICE)
    out0 = model(x0, patch_mask=m0)
    print(f"Sanity: x {tuple(x0.shape)} mask {tuple(m0.shape)} logits {tuple(out0.shape)}")

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer)
        metrics = evaluate(model, test_loader)

        print(f"\n[Epoch {epoch}] loss={loss:.4f} | "
              f"ICBHI={metrics['icbhi']:.4f} | SE={metrics['se']:.4f} | SP={metrics['sp']:.4f} | AUC={metrics['auc']:.4f}")
        print("Confusion Matrix:\n", metrics["cm"])

        if metrics["icbhi"] > best_icbhi:
            best_icbhi = metrics["icbhi"]
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model: {save_path} (ICBHI={best_icbhi:.4f})")

    # 最终详细报告
    final = evaluate(model, test_loader)
    print("\n=== Final Report ===")
    print(f"Threshold={BEST_THRESHOLD} | TOP_K={TOP_K} | WEIGHT_POS={WEIGHT_POS}")
    print(f"SE={final['se']:.4f} | SP={final['sp']:.4f} | ICBHI={(final['se']+final['sp'])/2:.4f} | AUC={final['auc']:.4f}")
    print("CM:\n", final["cm"])
    print(classification_report(final["y_true"], final["y_pred"], target_names=["Normal", "Abnormal"]))


if __name__ == "__main__":
    main()
