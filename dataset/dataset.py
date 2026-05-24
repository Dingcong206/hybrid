import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ICBHIASTTokenDataset(Dataset):
    """
    Dataset for ICBHI AST patch tokens.

    CSV should contain:
        tokens_path: path to .npy token file
        label: class label

    Each tokens.npy:
        [948, 768]

    Return:
        tokens: [948, 768]
        label: long tensor
    """

    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.transform = transform

        self.df = pd.read_csv(csv_path)

        if "tokens_path" not in self.df.columns:
            raise ValueError(f"{csv_path} 中没有 tokens_path 列")

        if "label" not in self.df.columns:
            raise ValueError(f"{csv_path} 中没有 label 列")

        self.df = self.df[self.df["tokens_path"].notna()].reset_index(drop=True)

        valid_indices = []
        print(f"[INFO] 正在检查 tokens 文件: {csv_path}")

        for i, row in self.df.iterrows():
            token_path = row["tokens_path"]
            if isinstance(token_path, str) and os.path.exists(token_path):
                valid_indices.append(i)

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        print(f"[INFO] 可用样本数: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        token_path = row["tokens_path"]
        label = int(row["label"])

        tokens = np.load(token_path)              # [948, 768]
        tokens = torch.from_numpy(tokens).float() # float32

        if self.transform is not None:
            tokens = self.transform(tokens)

        label = torch.tensor(label).long()

        return tokens, label