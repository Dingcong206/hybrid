import os
import glob
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 导入你的模型
from .model import SSA_Model, build_model

# --- 1. 配置 ---
PATCH_DIR = "/data/dingcong/hybrid/hear_patch_final"
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
CHECKPOINT_PATH = "your_model_checkpoint.pth"  # 你的模型权重路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LR 优化的超参数
TOP_K = 5
BEST_THRESHOLD = 0.80
BEST_WEIGHT = 3.0


def get_label(base_name):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path): return None
    df = pd.read_csv(txt_path, sep='\t', header=None)
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0


# --- 2. 初始化你的模型 ---
print("🚀 正在加载 SSA-Model...")
model = build_model()  # 或者 SSA_Model(...) 根据你的定义
# 如果有权重则加载
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- 3. 提取模型深层特征 ---
# 这一步是将 Patch 转化为模型认知的“高级语义向量”
print("🧠 正在通过模型提取深层特征...")
feat_files = sorted(glob.glob(os.path.join(PATCH_DIR, "*.npy")))
file_data = []

with torch.no_grad():
    for f_path in tqdm(feat_files):
        base_name = os.path.basename(f_path).replace(".npy", "")
        label = get_label(base_name)
        if label is None: continue

        # 加载 Patch 数据: 形状通常是 (N, seq_len, dim) 或类似
        patches = np.load(f_path)
        patches_torch = torch.from_numpy(patches).float().to(DEVICE)

        # 喂入模型获取特征 (例如取 cls_token 或 global_pool 的输出)
        # 假设你的模型 forward 返回的是 (N, hidden_dim)
        deep_features = model(patches_torch)

        if isinstance(deep_features, tuple):  # 兼容返回多个值的情况
            deep_features = deep_features[0]

        file_data.append({
            'name': base_name,
            'X': deep_features.cpu().numpy(),  # 模型输出的高级特征矩阵 (N, dim)
            'y': label
        })

# --- 4. 划分数据集 ---
train_data, test_data = train_test_split(file_data, test_size=0.2, random_state=42)

X_train = np.vstack([d['X'] for d in train_data])
y_train = np.hstack([[d['y']] * len(d['X']) for d in train_data])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 5. LR 线性探测 (Linear Probing) ---
print("⚖️ 正在进行 LR 线性探测评估...")
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1.0, 1: BEST_WEIGHT},
    C=0.001,
    solver='liblinear',
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

# --- 6. 验证与 Top-K 聚合 ---
y_true_file = []
y_pred_file = []

for d in test_data:
    X_test_scaled = scaler.transform(d['X'])
    probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    # 保持你之前的 Top-K=5 逻辑
    actual_k = min(TOP_K, len(probs))
    top_probs = np.sort(probs)[-actual_k:]
    score = np.mean(top_probs)

    pred = 1 if score >= BEST_THRESHOLD else 0
    y_true_file.append(d['y'])
    y_pred_file.append(pred)

# --- 7. 输出结果 ---
tn, fp, fn, tp = confusion_matrix(y_true_file, y_pred_file).ravel()
se = tp / (tp + fn) if (tp + fn) > 0 else 0
sp = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "=" * 50)
print(f"🌟 SSA-Model + LR Baseline 结果:")
print(f"SE: {se:.4f} | SP: {sp:.4f} | Score: {(se + sp) / 2:.4f}")
print("-" * 50)
print(classification_report(y_true_file, y_pred_file, target_names=['Normal', 'Abnormal']))