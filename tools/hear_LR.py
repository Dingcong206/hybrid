import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# --- 1. 配置与加载 ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official_baseline"
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
MIN_THRESHOLD = 0.45  # 你的底线要求

def get_label(base_name):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path): return None
    df = pd.read_csv(txt_path, sep='\t', header=None)
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0

X, y = [], []
feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
for f_path in feat_files:
    base_name = os.path.basename(f_path).replace(".npy", "")
    l = get_label(base_name)
    if l is not None:
        X.append(np.load(f_path).squeeze())
        y.append(l)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 2. 约束搜索循环 ---
weights = np.arange(1.0, 10.1, 0.2) # 扩大搜索范围到 10.0
valid_results = []

print("\n" + "="*85)
print(f"{'异常权重':>8} | {'SE (灵敏度)':>12} | {'SP (特异性)':>12} |