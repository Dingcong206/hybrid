#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ============================================================
# ✅ 你的参数：全部写死在这里（直接 python 运行）
# ============================================================
CONFIG = {
    # data
    "ROOT": "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens",  # 含 train_index.csv / test_index.csv

    # train
    "EPOCHS": 100,
    "BATCH_SIZE": 2,
    "ACCUM_STEPS": 8,

    "LR": 1e-5,
    "WEIGHT_DECAY": 1e-2,
    "NUM_WORKERS": 1,
    "SEED": 42,
    "DEVICE": "cuda",

    "SAVE_DIR": "/data/dingcong/hybrid/checkpoints_icbhi_4cls_like_author",  # 你可改回你自己的目录
    "PATIENCE": 15,

    # model
    "IN_DIM": 768,
    "D_MODEL": 512,
    "N_LAYERS": 8,
    "NHEAD": 4,
    "DROPOUT": 0.3,
    "MAX_LEN": 1024,

    # aug (默认关闭，与你现在一致)
    "SPEC_AUG": False,
    "MAX_MASK_T": 10,
    "MAX_MASK_F": 4,
    "NUM_MASKS": 2,

    # amp
    "AMP": True,

    # eval mode (作者代码里 two_cls_eval 的效果)
    # False: 严格四分类命中（pred==gt 才算对）
    # True : gt!=0 且 pred>0 算对（“二分类式命中统计”，但仍走 get_score）
    "TWO_CLS_EVAL": False,

    # weighted CE（你想开就改 True）
    "WEIGHTED_LOSS": False,
}

# ============================================================
# Path / import your backbone
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from mymodels import build_model  # 你的 SSA backbone：输出 (B, feat_dim)


# ============================================================
# 1) Seed
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2) SpecAugment（tokens 维度遮挡）
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
# 3) Dataset：读 tokens.npy（四分类 label 0/1/2/3）
# ============================================================
class TokenNPY4ClsDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        is_train: bool,
        specaug: bool,
        max_mask_t: int,
        max_mask_f: int,
        num_masks: int,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        if "tokens_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"[Dataset] CSV 必须包含 tokens_path / label 两列。当前列={self.df.columns.tolist()}")

        self.is_train = is_train
        self.specaug = specaug
        self.max_mask_t = max_mask_t
        self.max_mask_f = max_mask_f
        self.num_masks = num_masks

        self.y4 = self.df["label"].astype(int).values
        self.class_counts_4 = np.bincount(self.y4, minlength=4)

        print(
            f"[Dataset] Loaded {len(self.df)} samples from {csv_path} | "
            f"counts4={self.class_counts_4.tolist()} | "
            f"train={self.is_train} specaug={self.specaug}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["tokens_path"])
        x = torch.from_numpy(x).float()  # (T, D)

        y4 = int(self.y4[idx])

        if self.is_train and self.specaug:
            x = apply_spec_augment(x, self.max_mask_t, self.max_mask_f, self.num_masks)

        return x, torch.tensor(y4, dtype=torch.long)


# ============================================================
# 4) collate：pad + mask
# ============================================================
def collate_pad(batch):
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    D = xs[0].shape[1]
    T_max = max(lens)
    B = len(xs)

    x_pad = torch.zeros(B, T_max, D, dtype=torch.float32)
    mask = torch.ones(B, T_max, dtype=torch.bool)  # True=PAD

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        mask[i, :T] = False

    y = torch.stack(ys).view(-1)
    return x_pad, mask, y


# ============================================================
# 5) 作者式 get_score：hits/counts -> SP/SE/Score
# ============================================================
def get_score_from_hits_counts(hits: List[float], counts: List[float]) -> Tuple[float, float, float]:
    eps = 1e-10
    sp = 100.0 * (hits[0] / (counts[0] + eps))
    abn_hits = float(hits[1] + hits[2] + hits[3])
    abn_counts = float(counts[1] + counts[2] + counts[3])
    se = 100.0 * (abn_hits / (abn_counts + eps))
    score = (sp + se) / 2.0
    return float(sp), float(se), float(score)


@torch.no_grad()
def evaluate_like_author(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    two_cls_eval: bool,
) -> Dict[str, float]:
    backbone.eval()
    classifier.eval()

    hits = [0.0] * 4
    counts = [0.0] * 4

    all_true, all_pred = [], []
    ce_sum = 0.0
    n_sum = 0

    ce = nn.CrossEntropyLoss(reduction="sum").to(device)

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bsz = y.size(0)

        feat = backbone(x, mask=mask)         # (B, feat_dim)
        out = classifier(feat)               # (B, 4)
        loss = ce(out, y)

        ce_sum += float(loss.item())
        n_sum += int(bsz)

        pred = torch.argmax(out, dim=1)

        all_true.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())

        for i in range(bsz):
            gt = int(y[i].item())
            pr = int(pred[i].item())
            counts[gt] += 1.0

            if not two_cls_eval:
                if pr == gt:
                    hits[gt] += 1.0
            else:
                if gt == 0 and pr == 0:
                    hits[gt] += 1.0
                elif gt != 0 and pr > 0:
                    hits[gt] += 1.0

    sp, se, sc = get_score_from_hits_counts(hits, counts)

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "SP": sp, "SE": se, "ICBHI": sc,
        "ACC": float(acc), "F1": float(f1),
        "LOSS": float(ce_sum / max(1, n_sum)),
        "hits": hits, "counts": counts,
        "cm": cm,
    }


# ============================================================
# 6) Train one epoch
# ============================================================
def train_one_epoch(
    loader: DataLoader,
    backbone: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    accum_steps: int,
    scheduler=None,
    class_weights: torch.Tensor = None,
) -> float:
    backbone.train()
    classifier.train()

    if class_weights is not None:
        ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        ce = nn.CrossEntropyLoss()

    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    for i, (x, mask, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            feat = backbone(x, mask=mask)
            out = classifier(feat)
            loss = ce(out, y) / accum_steps

        scaler.scale(loss).backward()
        running += float(loss.item() * accum_steps)

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(classifier.parameters()), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

    # tail
    if len(loader) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(classifier.parameters()), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    return running / max(1, len(loader))


# ============================================================
# 7) Main
# ============================================================
def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])

    device = torch.device("cuda" if (cfg["DEVICE"] == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print("[DEBUG] device_count:", torch.cuda.device_count())
        print("[DEBUG] current_device:", torch.cuda.current_device())
        print("[DEBUG] device_name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    root = Path(cfg["ROOT"])
    train_csv = root / "train_index.csv"
    test_csv = root / "test_index.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"找不到：\n{train_csv}\n{test_csv}")

    os.makedirs(cfg["SAVE_DIR"], exist_ok=True)
    ckpt_best = os.path.join(cfg["SAVE_DIR"], "best.pth")
    ckpt_last = os.path.join(cfg["SAVE_DIR"], "last.pth")

    # ✅ 作者式：train 用 train_index.csv，val/test 用 test_index.csv
    train_ds = TokenNPY4ClsDataset(
        str(train_csv),
        is_train=True,
        specaug=cfg["SPEC_AUG"],
        max_mask_t=cfg["MAX_MASK_T"],
        max_mask_f=cfg["MAX_MASK_F"],
        num_masks=cfg["NUM_MASKS"],
    )
    val_ds = TokenNPY4ClsDataset(
        str(test_csv),
        is_train=False,
        specaug=False,
        max_mask_t=0,
        max_mask_f=0,
        num_masks=0,
    )

    dl_train = DataLoader(
        train_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_pad,
        drop_last=True,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_pad,
        drop_last=False,
    )

    # build backbone
    backbone = build_model(
        in_dim=cfg["IN_DIM"],
        d_model=cfg["D_MODEL"],
        n_layers=cfg["N_LAYERS"],
        nhead=cfg["NHEAD"],
        dropout=cfg["DROPOUT"],
        max_len=cfg["MAX_LEN"],
        num_classes=4,
    ).to(device)

    if not hasattr(backbone, "final_feat_dim"):
        raise RuntimeError("你的 build_model 返回的 backbone 没有 final_feat_dim，无法外接分类头。")

    # 4-class head
    classifier = nn.Linear(backbone.final_feat_dim, 4).to(device)

    # weighted CE（可选）
    class_weights = None
    if cfg["WEIGHTED_LOSS"]:
        counts = train_ds.class_counts_4.astype(np.float32)
        w = 1.0 / np.maximum(counts, 1.0)
        w = w / w.sum() * 4.0
        class_weights = torch.tensor(w, dtype=torch.float32)
        print(f"[INFO] weighted_loss ON. class_counts={counts.tolist()} weights={w.tolist()}")

    # optimizer
    params = list(backbone.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"], betas=(0.9, 0.999))

    # scheduler：按真实 optimizer step 数（考虑 accum）
    accum_steps = int(cfg["ACCUM_STEPS"])
    opt_steps_per_epoch = math.ceil(len(dl_train) / accum_steps)
    total_opt_steps = opt_steps_per_epoch * int(cfg["EPOCHS"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_opt_steps))

    # AMP
    use_amp = bool(cfg["AMP"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    best_epoch = -1
    bad_epochs = 0

    print("\n🚀 Start training (4-class CE + author-style get_score)\n")

    for epoch in range(1, int(cfg["EPOCHS"]) + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            loader=dl_train,
            backbone=backbone,
            classifier=classifier,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            accum_steps=accum_steps,
            scheduler=scheduler,
            class_weights=class_weights,
        )

        val_m = evaluate_like_author(
            loader=dl_val,
            backbone=backbone,
            classifier=classifier,
            device=device,
            two_cls_eval=bool(cfg["TWO_CLS_EVAL"]),
        )

        score = val_m["ICBHI"]
        improved = score > best_score + 1e-9

        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": backbone.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "best_score": best_score,
                    "config": deepcopy(cfg),
                },
                ckpt_best
            )
            star = "⭐"
        else:
            bad_epochs += 1
            star = " "

        # save last
        torch.save(
            {
                "epoch": epoch,
                "backbone_state": backbone.state_dict(),
                "classifier_state": classifier.state_dict(),
                "best_score": best_score,
                "config": deepcopy(cfg),
            },
            ckpt_last
        )

        dt = time.time() - t0
        print(
            f"{star} Epoch {epoch:03d}/{cfg['EPOCHS']} | "
            f"train_loss {train_loss:.4f} | "
            f"VAL Score {val_m['ICBHI']:.4f} SP {val_m['SP']:.4f} SE {val_m['SE']:.4f} | "
            f"ACC {val_m['ACC']:.4f} F1 {val_m['F1']:.4f} | "
            f"{dt:.1f}s"
        )

        if bad_epochs >= int(cfg["PATIENCE"]):
            print(f"[EARLY STOP] Score 连续 {cfg['PATIENCE']} 轮无提升，停止于 epoch {epoch}（best@{best_epoch} Score={best_score:.4f}）")
            break

    print(f"\n✅ DONE. Best Score={best_score:.4f} @ epoch {best_epoch}")
    print(f"[SAVED] best checkpoint: {ckpt_best}")

    # final eval from best
    print("\n🚀 Final evaluation from best checkpoint (same author-style get_score)\n")
    ckpt = torch.load(ckpt_best, map_location=device)
    backbone.load_state_dict(ckpt["backbone_state"])
    classifier.load_state_dict(ckpt["classifier_state"])

    final_m = evaluate_like_author(
        loader=dl_val,
        backbone=backbone,
        classifier=classifier,
        device=device,
        two_cls_eval=bool(cfg["TWO_CLS_EVAL"]),
    )
    print(
        f"[FINAL] Score {final_m['ICBHI']:.4f} SP {final_m['SP']:.4f} SE {final_m['SE']:.4f} | "
        f"ACC {final_m['ACC']:.4f} F1 {final_m['F1']:.4f}"
    )
    print("[FINAL] Confusion Matrix (rows=true 0/1/2/3, cols=pred 0/1/2/3):")
    print(final_m["cm"])


if __name__ == "__main__":
    main()
