import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RespiratoryDataset(Dataset):
    def __init__(self, csv_path, feat_dir, transform=None):
        """
        Args:
            csv_path (str): 标签 CSV 文件的路径
            feat_dir (str): 存放 .npy 特征文件的目录路径
            transform (callable, optional): 可选的特征增强逻辑
        """
        # 1. 读取 CSV 文件
        self.df = pd.read_csv(csv_path)
        self.feat_dir = feat_dir
        self.transform = transform

        # 2. 检查文件是否存在，防止训练中断
        # 我们只保留那些在文件夹里真实存在的特征文件
        self.valid_indices = []
        print("正在检查特征文件完整性...")
        for i, row in self.df.iterrows():
            f_path = os.path.join(self.feat_dir, row['file_name'])
            if os.path.exists(f_path):
                self.valid_indices.append(i)

        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"✅ 检查完成！可用样本数: {len(self.df)}")

    def __len__(self):
        # 返回数据集的总大小
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 获取对应的文件名和标签
        row = self.df.iloc[idx]
        file_name = row['file_name']
        label = row['label']

        # 2. 从硬盘加载 .npy 特征
        feat_path = os.path.join(self.feat_dir, file_name)
        # mmap_mode='r' 是一种高级优化：它只在需要时读取数据，不占用过多内存
        feat = np.load(feat_path, mmap_mode='r')

        # 3. 转换为 PyTorch 张量 (Tensor)
        # 原始特征是 (32768, 256)，转换为 float32 格式
        feat_tensor = torch.from_numpy(np.array(feat)).float()
        label_tensor = torch.tensor(label).float()  # 对于 BCE 损失，标签通常需要 float

        # 4. 如果有数据增强（可选）
        if self.transform:
            feat_tensor = self.transform(feat_tensor)

        return feat_tensor, label_tensor