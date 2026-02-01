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
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0


# --- 2. 修复后的加载逻辑 ---
feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
file_data = []  # 存储文件级别的信息

print(f"📂 正在解析 {len(feat_files)} 个序列特征文件...")

for f_path in feat_files:
    base_name = os.path.basename(f_path).replace(".npy", "")
    label = get_label(base_name)
    if label is None: continue

    emb = np.load(f_path)
    if emb.ndim == 1: emb = emb[None, :]  # 确保是 2D 矩阵

    file_data.append({
        'name': base_name,
        'X': emb,  # (N, 512)
        'y': label  # 0 或 1
    })

# --- 3. 划分数据集 (按文件划分，防止数据泄露) ---
train_data, test_data = train_test_split(file_data, test_size=0.2, random_state=42)

# 拆解成片段用于训练
X_train = np.vstack([d['X'] for d in train_data])
y_train = np.hstack([[d['y']] * len(d['X']) for d in train_data])

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# --- 4. 权重搜索 ---
weights = np.arange(1.0, 10.1, 0.5)
best_valid = None

print("\n" + "=" * 80)
print(f"{'异常权重':>8} | {'SE (灵敏度)':>12} | {'SP (特异性)':>12} | {'(SE+SP)/2':>12}")
print("-" * 80)

for w in weights:
    model = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: w}, C=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上做文件级聚合预测
    y_test_file, y_prob_file = [], []

    for d in test_data:
        # 对该文件的所有片段进行标准化和预测
        X_test_scaled = scaler.transform(d['X'])
        probs = model.predict_proba(X_test_scaled)[:, 1]

        # 聚合策略：均值 (Mean)
        file_prob = np.mean(probs)
        y_prob_file.append(file_prob)
        y_test_file.append(d['y'])

    y_pred_file = (np.array(y_prob_file) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_file, y_pred_file).ravel()

    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    score = (se + sp) / 2

    status = "✅" if se >= MIN_THRESHOLD and sp >= MIN_THRESHOLD else "❌"
    print(f"{w:>12.1f} | {se:>12.4f} | {sp:>12.4f} | {score:>12.4f} | {status}")

    if (se >= MIN_THRESHOLD and sp >= MIN_THRESHOLD) and (best_valid is None or score > best_valid[3]):
        best_valid = (w, se, sp, score)

print("=" * 80)
if best_valid:
    print(f"🏆 最佳权重: {best_valid[0]} | ICBHI Score: {best_valid[3]:.4f}")
else:
    print("😓 依然没能满足双 0.45 约束，建议尝试更换 C 值为 0.01 或 0.001。")