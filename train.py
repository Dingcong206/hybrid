import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# 导入你刚才准备好的模型文件
from VimA_Model import VimAHybrid

# ================= 配置区 (已适配 Linux 路径) =================
# 获取当前脚本所在的文件夹路径 (即 /data/dingcong/hybrid)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
NPY_DIR = os.path.join(BASE_DIR, "spec_npy_v2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# 正负样本权重 (根据你的混淆矩阵估算，标签 0 (正常) 有 62个，标签 1 (异常) 有 122个)
# 如果想让模型更关注标签 0 (少数类)，可以增加其权重，或者降低标签 1 的权重。
# pos_weight = 负样本数量 / 正样本数量 = 62 / 122 = 0.508
# 推荐从 1.0 (默认) 调整到 0.5 ~ 0.8 之间，让模型对标签 1 的预测不那么激进
POS_WEIGHT_VALUE = 0.7  # 从 1.0 开始尝试，然后降到 0.8, 0.7 观察效果


# =============================================================

# 1. 定义数据加载工具
class ICBHIDataset(torch.utils.data.Dataset):
    def __init__(self, df, npy_dir):
        self.df = df
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 确保文件名对应
        npy_path = os.path.join(self.npy_dir, row['wav_name'].replace('.wav', '.npy'))

        # 检查文件是否存在
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"在 {self.npy_dir} 中找不到文件: {npy_path}")

        # 加载数据
        spec = np.load(npy_path)  # (128, 1024)
        spec_t = torch.from_numpy(spec).float().unsqueeze(0)  # 变为 (1, 128, 1024)
        label = torch.tensor(row['label'], dtype=torch.float)
        return spec_t, label


# 2. 准备数据
print("正在加载数据索引...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"找不到 CSV 文件: {CSV_PATH}. 请确认路径和文件名是否正确。")
df = pd.read_csv(CSV_PATH)

# 确保标签列存在且数据类型正确
if 'label' not in df.columns:
    raise ValueError("CSV 文件中缺少 'label' 列。")
if 'wav_name' not in df.columns:
    raise ValueError("CSV 文件中缺少 'wav_name' 列。")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42,
                                    stratify=df['label'])  # 添加 stratify 确保训练集和验证集的标签分布相似

train_loader = DataLoader(ICBHIDataset(train_df, NPY_DIR), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4)  # num_workers 可根据服务器CPU核数调整
val_loader = DataLoader(ICBHIDataset(val_df, NPY_DIR), batch_size=BATCH_SIZE, num_workers=4)

# 3. 初始化模型、损失函数和优化器
print(f"正在 {DEVICE} 上初始化 VimA 模型...")
model = VimAHybrid(num_classes=1, d_model=192, patch_time=1).to(DEVICE)

# 为 BCEWithLogitsLoss 设置正样本权重
pos_weight_tensor = torch.tensor([POS_WEIGHT_VALUE]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
# 学习率调度器，让学习率在后期平滑下降
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 4. 训练循环
print("开始训练...")
best_icbhi_score = 0
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for specs, labels in train_loader:
        specs, labels = specs.to(DEVICE), labels.to(DEVICE)

        # 前向传播
        outputs = model(specs)  # outputs 形状为 [Batch]
        loss = criterion(outputs, labels)  # labels 形状也为 [Batch]

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()  # 每个 Epoch 结束时更新学习率

    # 验证环节
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)

            logits = model(specs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()  # 默认阈值 0.5

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算各项指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    # 混淆矩阵的四个元素：(tn, fp, fn, tp)
    cm_array = confusion_matrix(all_labels, all_preds).ravel()
    # 确保 cm_array 有四个元素，否则可能出现类别不全的情况
    if len(cm_array) == 4:
        tn, fp, fn, tp = cm_array
    else:  # 只有一类被预测到或只有一类真实存在
        # 此时需要根据实际情况手动判断，这里给出简化处理
        print("警告: 混淆矩阵不完整，可能只有单一类别被预测或存在。")
        # 假设真实标签中包含两类
        if len(set(all_labels)) == 2:
            if 0 not in all_preds:  # 全部预测为1
                tn, fp, fn, tp = 0, len([l for l in all_labels if l == 0]), len([l for l in all_labels if l == 1]), 0
            elif 1 not in all_preds:  # 全部预测为0
                tn, fp, fn, tp = len([l for l in all_labels if l == 0]), 0, 0, len([l for l in all_labels if l == 1])
            else:  # 其他情况，可能需要更复杂的处理
                tn, fp, fn, tp = 0, 0, 0, 0  # 暂时置0，避免后续计算错误
        else:  # 真实标签只有一类
            tn, fp, fn, tp = 0, 0, 0, 0  # 暂时置0

    se = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度（真阳率）
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度（真阴率）
    icbhi_score = (se + sp) / 2  # ICBHI 官方分数

    print(f"\n--- Epoch [{epoch + 1}/{EPOCHS}] ---")
    print(f"Train Loss: {train_loss / len(train_loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Validation: ACC: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"            SE: {se:.4f} | SP: {sp:.4f} | ICBHI Score: {icbhi_score:.4f}")

    # 保存最优模型 (现在根据 ICBHI Score)
    if icbhi_score > best_icbhi_score:
        best_icbhi_score = icbhi_score
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "best_vima_model.pth")
        print(f"⭐ 发现更高 ICBHI Score ({best_icbhi_score:.4f}) 的模型，已保存权重！")

    # 在最后一个 Epoch 绘制并保存混淆矩阵
    if epoch == EPOCHS - 1:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal (0)', 'Abnormal (1)'],
                    yticklabels=['Normal (0)', 'Abnormal (1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.savefig('confusion_matrix_final.png')
        print("📊 最终混淆矩阵图已保存至 confusion_matrix_final.png")

print(f"\n✅ 训练完成！最高 ICBHI Score: {best_icbhi_score:.4f} (Epoch {best_epoch})")