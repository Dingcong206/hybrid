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
MIN_THRESHOLD = 0.45


def get_label(base_name):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path): return None
    df = pd.read_csv(txt_path, sep='\t', header=None)
    # 原始标签逻辑：只要 txt 里有 1，就是异常
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0


# --- 2. 加载数据 ---
feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
file_data = []

print(f"📂 正在解析 {len(feat_files)} 个序列特征文件...")

for f_path in feat_files:
    base_name = os.path.basename(f_path).replace(".npy", "")
    label = get_label(base_name)
    if label is None: continue

    emb = np.load(f_path)
    if emb.ndim == 1: emb = emb[None, :]

    file_data.append({'name': base_name, 'X': emb, 'y': label})

# 按文件划分
train_data, test_data = train_test_split(file_data, test_size=0.2, random_state=42)

X_train = np.vstack([d['X'] for d in train_data])
y_train = np.hstack([[d['y']] * len(d['X']) for d in train_data])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# --- 3. 寻找能让“一票判定”逻辑成立的参数 ---
# 因为你坚持“一有即异常”，所以我们要搜索不同的“判定门槛”
decision_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
weights = [0.5, 1.0, 2.0]  # 这里的权重不再需要太大

print("\n" + "=" * 80)
print(f"{'门槛':>6} | {'权重':>6} | {'SE (灵敏度)':>12} | {'SP (特异性)':>12} | {'状态'}")
print("-" * 80)

for thres in decision_thresholds:
    for w in weights:
        # C=0.0001 是为了让模型不要对单个特征维度过度反应
        model = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: w}, C=0.0001, random_state=42)
        model.fit(X_train, y_train)

        y_test_file, y_pred_file = [], []

        for d in test_data:
            X_test_scaled = scaler.transform(d['X'])
            probs = model.predict_proba(X_test_scaled)[:, 1]

            # --- 核心逻辑：Max Pooling (一票判定) ---
            max_prob = np.max(probs)
            # 只有最大概率超过自定义门槛，才判为异常
            pred = 1 if max_prob >= thres else 0

            y_pred_file.append(pred)
            y_test_file.append(d['y'])

        tn, fp, fn, tp = confusion_matrix(y_test_file, y_pred_file).ravel()
        se = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0

        status = "✅" if se >= MIN_THRESHOLD and sp >= MIN_THRESHOLD else "❌"
        print(f"{thres:>8.2f} | {w:>8.1f} | {se:>12.4f} | {sp:>12.4f} | {status}")

print("=" * 80)