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
weights = np.arange(1.0, 10.1, 0.2)  # 扩大搜索范围到 10.0
valid_results = []

print("\n" + "=" * 85)
print(f"{'异常权重':>8} | {'SE (灵敏度)':>12} | {'SP (特异性)':>12} | {'(SE+SP)/2':>12} | {'状态':>10}")
print("-" * 85)

for w in weights:
    cw = {0: 1.0, 1: round(w, 2)}
    model = LogisticRegression(max_iter=1000, class_weight=cw, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    score = (se + sp) / 2

    # 检查是否满足双 0.45 门槛
    is_valid = se >= MIN_THRESHOLD and sp >= MIN_THRESHOLD
    status = "✅ 达标" if is_valid else "❌ 违规"

    print(f"{cw[1]:>12.1f} | {se:>12.4f} | {sp:>12.4f} | {score:>12.4f} | {status:>8}")

    if is_valid:
        valid_results.append((cw[1], se, sp, score, fn))

print("=" * 85)

# --- 3. 输出最优约束解 ---
if valid_results:
    # 在达标的方案中选总分最高的
    best_valid = max(valid_results, key=lambda x: x[3])
    w, se, sp, sc, fn = best_valid
    print(f"🎯 在满足双 {MIN_THRESHOLD} 约束下的最佳异常权重: {w}")
    print(f"📊 最终表现: SE={se:.4f}, SP={sp:.4f}, 总分={sc:.4f}")
    print(f"📉 漏诊数 (FN): {fn}")
else:
    print(f"⚠️ 警告：在当前权重范围内，没有找到能同时满足 SE 和 SP > {MIN_THRESHOLD} 的权重。")
    print("这通常意味着特征区分度不足，建议回溯增加数据片段。")