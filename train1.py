import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# --- 核心引用：从你的 SSA_Model.py 导入模型 ---
# 确保 SSA_Model.py 和此脚本在同一目录下
from SSA_Model import SSA_Model


# =====================================================
# 1) Dataset 类：处理 (97, 1024) 特征
# =====================================================
class ICBHIDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 加载 HeAR 提取的特征 [97, 1024]
        feature = np.load(row['feature_path']).astype(np.float32)
        label = int(row['label'])
        return torch.from_numpy(feature), torch.tensor(label, dtype=torch.float32)


# =====================================================
# 2) 主训练程序
# =====================================================
def main():
    # --- 参数配置 ---
    CSV_PATH = "/data/dingcong/hybrid/metadata_segmented.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-4
    df = pd.read_csv(CSV_PATH)

    print("📊 Label 分布：")
    print(df["label"].value_counts())
    print("\n📈 正负样本比例：")
    print(df["label"].value_counts(normalize=True))

    # --- 数据准备 (按 Patient ID 划分) ---
    df = pd.read_csv(CSV_PATH)
    # 提取 ID 前缀，确保同一个病人的数据不跨集
    id_col = "original_wav" if "original_wav" in df.columns else "user_id"
    df["patient_id"] = df[id_col].apply(lambda x: str(x).split('_')[0])
    patient_label = df.groupby("patient_id")["label"].max()
    print("👤 病人级别 label 分布：")
    print(patient_label.value_counts())
    unique_patients = df["patient_id"].unique()
    train_p, val_p = train_test_split(unique_patients, test_size=0.2, random_state=42)

    train_loader = DataLoader(ICBHIDataset(df[df["patient_id"].isin(train_p)]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ICBHIDataset(df[df["patient_id"].isin(val_p)]), batch_size=BATCH_SIZE, shuffle=False)

    print(f"✅ 划分完成: 训练集 {len(train_p)} 人 | 验证集 {len(val_p)} 人")

    # --- 模型初始化 (input_dim 设为 1024) ---
    model = SSA_Model(input_dim=1024, d_model=256, n_layers=6).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 训练循环 ---
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # 验证
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                outputs = model(feats)
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        scheduler.step()

        # 计算指标
        preds = [1 if p > 0.5 else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)

        print(f"Epoch {epoch + 1} | Loss: {t_loss / len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_ssa_mamba.pth")
            print("⭐ 发现更好模型，已保存权重")


if __name__ == "__main__":
    main()