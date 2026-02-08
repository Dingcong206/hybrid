import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# 1. Dataset 类：读取你预处理生成的 npy 和 csv
# ============================================================
class ICBHINpyDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # 加载预处理好的 (798, 128) fbank
        fbank = np.load(row['fbank_path'])
        label = int(row['label'])

        # AST模型通常期望形状为 [798, 128]
        # 如果你用的是 CNN 模型 (ResNet)，请使用:
        # fbank = torch.from_numpy(fbank).transpose(0, 1).unsqueeze(0)
        fbank = torch.from_numpy(fbank).float()
        return fbank, label


# ============================================================
# 2. 核心评估函数：官方 Sp, Se, Sc 逻辑
# ============================================================
def get_icbhi_scores(preds, labels):
    # 0:Normal, 1:Crackle, 2:Wheeze, 3:Both
    hits = [0.0] * 4
    counts = [0.0] * 4
    for p, l in zip(preds, labels):
        counts[l] += 1
        if p == l:
            hits[l] += 1

    # Specificity (正常类的召回率)
    sp = (hits[0] / (counts[0] + 1e-10)) * 100
    # Sensitivity (异常类 1,2,3 的总召回率)
    se_hits = sum(hits[1:])
    se_counts = sum(counts[1:])
    se = (se_hits / (se_counts + 1e-10)) * 100
    # 最终官方得分
    sc = (sp + se) / 2.0
    return sp, se, sc


# ============================================================
# 3. 训练主程序
# ============================================================
def main():
    # --- 配置参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 100

    TRAIN_CSV = "/data/dingcong/hybrid/icbhi_official_fbank/train_index.csv"
    TEST_CSV = "/data/dingcong/hybrid/icbhi_official_fbank/test_index.csv"
    SAVE_PATH = "best_official_score_model.pth"

    # --- 数据加载 ---
    train_dataset = ICBHINpyDataset(TRAIN_CSV)
    test_dataset = ICBHINpyDataset(TEST_CSV)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 模型初始化 ---
    # 这里请替换为你自己的模型定义，例如 AST 或 ResNet
    # 示例: model = YourModel(num_classes=4).to(DEVICE)
    # model = ASTModel.from_pretrained(...)
    # model.to(DEVICE)
    print(f"[INFO] 训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")

    # --- 关键：类别加权损失函数 ---
    # 根据你提供的统计结果：0(2063), 1(1215), 2(501), 3(363)
    # 使用倒数加权来平衡不均衡的数据集
    counts = torch.tensor([2063, 1215, 501, 363], dtype=torch.float)
    weights = (1.0 / counts) / (1.0 / counts).sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_sc = 0.0

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for fbanks, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            fbanks, labels = fbanks.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(fbanks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # --- 测试集评估 ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for fbanks, labels in test_loader:
                fbanks = fbanks.to(DEVICE)
                outputs = model(fbanks)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        sp, se, sc = get_icbhi_scores(all_preds, all_labels)

        print(
            f"Epoch [{epoch + 1}] Loss: {train_loss / len(train_loader):.4f} | Sp: {sp:.2f}% | Se: {se:.2f}% | Score: {sc:.2f}%")

        # --- 核心：按官方 Score 保存模型 ---
        if sc > best_sc:
            best_sc = sc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f">>> 发现更高分: {sc:.2f}%, 模型已保存至 {SAVE_PATH}")

    print(f"\n[DONE] 训练完成，最佳官方分 Sc: {best_sc:.2f}%")


if __name__ == "__main__":
    main()