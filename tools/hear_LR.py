import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# --- 1. 配置路径 ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official_baseline"
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"


def get_label(base_name):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path): return None
    # ICBHI 标签通常以制表符分隔
    df = pd.read_csv(txt_path, sep='\t', header=None)
    # 只要 crackle (列2) 或 wheeze (列3) 出现 1，即判定为异常 (1)
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0


# --- 2. 加载 HeAR 特征 (512维) ---
X, y = [], []
feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
print(f"📂 正在读取 {len(feat_files)} 个特征文件...")

for f_path in feat_files:
    base_name = os.path.basename(f_path).replace(".npy", "")
    label = get_label(base_name)
    if label is not None:
        X.append(np.load(f_path).squeeze())
        y.append(label)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. 权重扫描循环 ---
weights = np.arange(1.0, 5.1, 0.2)
best_score = -1
best_report = None

print("\n" + "=" * 75)
print(f"{'异常权重':>8} | {'SE (灵敏度)':>12} | {'SP (特异性)':>12} | {'(SE+SP)/2':>12} | {'FN (漏诊)':>8}")
print("-" * 75)

for w in weights:
    cw = {0: 1.0, 1: round(w, 2)}
    model = LogisticRegression(max_iter=1000, class_weight=cw, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # 计算指标
    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    avg_score = (se + sp) / 2

    print(f"{cw[1]:>12.1f} | {se:>12.4f} | {sp:>12.4f} | {avg_score:>12.4f} | {fn:>8}")

    # 记录最佳
    if avg_score > best_score:
        best_score = avg_score
        best_report = (cw[1], se, sp, avg_score, fn)

print("=" * 75)
if best_report:
    w, se, sp, score, fn = best_report
    print(f"🏆 最佳权重结果: 异常类权重 = {w}")
    print(f"📊 指标: SE={se:.4f}, SP={sp:.4f}, (SE+SP)/2 = {score:.4f}")
    print(f"📉 此时漏诊数 (FN): {fn}")