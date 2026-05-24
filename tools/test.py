import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =====================================================
# 1. 加入项目根目录
# =====================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataset.dataloader import build_dataloader
from mymodels.model import TimeFrequencyEncoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # 2. 这里写你生成 tokens 的 CSV 路径
    # =====================================================
    train_csv = "/data/dingcong/hybrid/icbhi_official_ast_patch_tokens/train_index.csv"

    print("[INFO] train_csv:", train_csv)

    # =====================================================
    # 3. 先检查 CSV
    # =====================================================
    df = pd.read_csv(train_csv)

    print("[INFO] CSV columns:", df.columns.tolist())
    print("[INFO] CSV samples:", len(df))
    print("[INFO] label counts:")
    print(df["label"].value_counts().sort_index())

    print("[INFO] first tokens_path:", df.iloc[0]["tokens_path"])
    print("[INFO] first tokens_shape in csv:", df.iloc[0].get("tokens_shape", "No tokens_shape column"))

    # =====================================================
    # 4. 直接读取第一个 npy 看 shape
    # =====================================================
    first_token_path = df.iloc[0]["tokens_path"]
    tokens_np = np.load(first_token_path)

    print("[INFO] first npy shape:", tokens_np.shape)
    print("[INFO] first npy dtype:", tokens_np.dtype)

    if tokens_np.shape != (948, 768):
        raise ValueError(
            f"tokens shape 不对，当前是 {tokens_np.shape}，期望是 (948, 768)。"
            f"请检查 AST patch token 生成过程。"
        )

    # =====================================================
    # 5. 测试 DataLoader
    # =====================================================
    loader = build_dataloader(
        csv_path=train_csv,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    tokens, labels = next(iter(loader))

    print("[INFO] dataloader tokens shape:", tokens.shape)
    print("[INFO] dataloader labels shape:", labels.shape)
    print("[INFO] dataloader labels:", labels)

    if tokens.shape[1:] != (948, 768):
        raise ValueError(
            f"DataLoader 输出 shape 不对，当前是 {tokens.shape}，"
            f"期望是 [B, 948, 768]。"
        )

    # =====================================================
    # 6. 测试 TimeFrequencyEncoder
    # =====================================================
    encoder = TimeFrequencyEncoder(
        token_dim=768,
        freq_patches=12,
        time_patches=79,
        time_depth=2,
        freq_depth=2,
        num_heads=8,
        dropout=0.1
    ).to(device)

    # 逻辑回归分类器
    classifier = nn.Linear(768, 4).to(device)

    encoder.eval()
    classifier.eval()

    tokens = tokens.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        features = encoder(tokens)        # [B, 768]
        logits = classifier(features)     # [B, 4]

    print("[INFO] encoder feature shape:", features.shape)
    print("[INFO] logits shape:", logits.shape)

    if features.shape != (tokens.shape[0], 768):
        raise ValueError(f"feature shape 不对: {features.shape}")

    if logits.shape != (tokens.shape[0], 4):
        raise ValueError(f"logits shape 不对: {logits.shape}")

    print("\n[SUCCESS] dataset → dataloader → encoder → logistic classifier 全部跑通！")


if __name__ == "__main__":
    main()