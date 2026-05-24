import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

print("[DEBUG] PROJECT_ROOT:", PROJECT_ROOT)

from dataset.dataloader import build_dataloader
from mymodels.model import TimeFrequencyEncoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/train_index.csv"

    print("[INFO] train_csv:", train_csv)

    df = pd.read_csv(train_csv)
    print("[INFO] CSV columns:", df.columns.tolist())
    print("[INFO] samples:", len(df))
    print("[INFO] first tokens_path:", df.iloc[0]["tokens_path"])

    tokens_np = np.load(df.iloc[0]["tokens_path"])
    print("[INFO] first npy shape:", tokens_np.shape)

    loader = build_dataloader(
        csv_path=train_csv,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    tokens, labels = next(iter(loader))

    print("[INFO] dataloader tokens:", tokens.shape)
    print("[INFO] labels:", labels.shape)

    encoder = TimeFrequencyEncoder(
        token_dim=768,
        freq_patches=12,
        time_patches=79,
        time_depth=2,
        freq_depth=2,
        num_heads=8,
        dropout=0.1
    ).to(device)

    classifier = nn.Linear(768, 4).to(device)

    tokens = tokens.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        features = encoder(tokens)
        logits = classifier(features)

    print("[INFO] features:", features.shape)
    print("[INFO] logits:", logits.shape)

    print("\n[SUCCESS] 全部跑通！")


if __name__ == "__main__":
    main()