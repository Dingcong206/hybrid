#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataloader import build_dataloader
from mymodels.model import TimeFrequencyEncoder


class TimeFrequencyLogisticModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.encoder = TimeFrequencyEncoder(
            token_dim=768,
            freq_patches=12,
            time_patches=79,
            time_depth=2,
            freq_depth=2,
            num_heads=8,
            dropout=0.1
        )

        # 逻辑回归分类器
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        feature = self.encoder(x)          # [B, 768]
        logits = self.classifier(feature)  # [B, 4]
        return logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model(tokens)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN_CSV = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/train_index.csv"
    TEST_CSV = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/test_index.csv"
    SAVE_PATH = "/data/dingcong/hybrid/best_token_model.pth"

    EPOCHS = 1
    BATCH_SIZE = 2
    LR = 5e-5

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    print("[INFO] device:", device)

    train_loader = build_dataloader(
        csv_path=TRAIN_CSV,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = build_dataloader(
        csv_path=TEST_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = TimeFrequencyLogisticModel(num_classes=4).to(device)

    # 类别权重
    train_df = pd.read_csv(TRAIN_CSV)
    counts = torch.zeros(4, dtype=torch.float32)
    for k, v in train_df["label"].value_counts().items():
        counts[int(k)] = float(v)

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 4.0

    print("[INFO] class counts:", counts.tolist())
    print("[INFO] class weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for tokens, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(tokens)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch
                },
                SAVE_PATH
            )
            print(f"[SAVE] best model saved to {SAVE_PATH}")

    print("[DONE] training finished.")
    print("[DONE] best acc:", best_acc)


if __name__ == "__main__":
    main()