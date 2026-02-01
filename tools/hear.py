import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CoswaraDataset(Dataset):
    def __init__(self, csv_file, feat_dir):
        self.df = pd.read_csv(csv_file)
        self.feat_dir = feat_dir

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = np.load(f"{self.feat_dir}/{row['user_id']}.npy")
        label = torch.tensor(row['label'], dtype=torch.float32)
        return torch.from_numpy(feat).float(), label


# 官方标准的 Linear Probing 分类器
class HeAR_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # 二分类：阴性/阳性
        )

    def forward(self, x): return self.net(x).squeeze()


# 训练逻辑
def train():
    dataset = CoswaraDataset("/data/dingcong/hybrid/Coswara-Data/combined_data.csv",
                             "/data/dingcong/hybrid/Coswara-Data/official_features")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = HeAR_Classifier().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x.cuda())
            loss = criterion(pred, y.cuda())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()