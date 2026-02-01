import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# --- 1. 配置 ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official"
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
TOP_K = 5
MIN_THRESHOLD = 0.45

def get_label(base_name):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path): return None
    df = pd.read_csv(txt_path, sep='\t', header=None)
    # 只要包含 Crackle 或 Wheeze 标记即为 1
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0

# --- 2. 加载数据 ---
feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
file_data = []
print(f"📂 正在加载 {len(feat_files)} 个序列特征文件...")

for f_path in feat_files:
    base_name = os.path.basename(f_path).replace(".npy", "")
    label = get_label(base_name)
    if label is None: continue
    emb = np.load(f_path)
    if emb.ndim == 1: emb = emb[None, :]
    file_data.append({'X': emb, 'y': label})

# 划分训练/测试
train_data, test_data = train_test_split(file_data, test_size=0.2, random_state=42)

X_train = np.vstack([d['X'] for d in train_data])
y_train = np.hstack([[d['y']] * len(d['X']) for d in train_data])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# --- 3. 固定 Top-K=5，搜索判定门槛和权重 ---
# 我们重点搜索不同的判定门槛 (Threshold)，因为这决定了 SP 能否回升
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
weights = [1.0, 1.5, 2.0, 3.0]

print("\n" + "="*85)
print(f"{'判定门槛':>8