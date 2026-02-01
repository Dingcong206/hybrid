import os
import glob
import numpy as np
import pandas as pd
import joblib  # 用于保存模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# --- 1. 最终配置 (基于实验得出的最优值) ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official"
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"
MODEL_SAVE_PATH = "hear_lr_baseline_model.pkl"
SCALER_SAVE_PATH = "hear_scaler_baseline.pkl"

TOP_K = 5
BEST_THRESHOLD = 0.80
BEST_WEIGHT = 3.0


def get_label(base_name):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path): return None
    df = pd.read_csv(txt_path, sep='\t', header=None)
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0


# --- 2. 加载数据 ---
feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
file_data = []

for f_path in feat_files:
    base_name = os.path.basename(f_path).replace(".npy", "")
    label = get_label(base_name)
    if label is None: continue
    emb = np.load(f_path)
    if emb.ndim == 1: emb = emb[None, :]
    file_data.append({'name': base_name, 'X': emb, 'y': label})

train_data, test_data = train_test_split(file_data, test_size=0.2, random_state=42)

X_train = np.vstack([d['X'] for d in train_data])
y_train = np.hstack([[d['y']] * len(d['X']) for d in train_data])

# --- 3. 训练与保存 ---
print("⚙️ 正在训练最终模型...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 使用最优参数
model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1.0, 1: BEST_WEIGHT},
    C=0.001,
    solver='liblinear',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 保存模型组件
joblib.dump(model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"💾 模型已保存至: {MODEL_SAVE_PATH}")

# --- 4. 详细评估 ---
print("\n📊 正在生成详细报告...")
results = []
y_true_file = []
y_pred_file = []
y_prob_file = []

for d in test_data:
    X_test_scaled = scaler.transform(d['X'])
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Top-K 聚合逻辑
    actual_k = min(TOP_K, len(probs))
    top_probs = np.sort(probs)[-actual_k:]
    mean_top_prob = np.mean(top_probs)

    pred = 1 if mean_top_prob >= BEST_THRESHOLD else 0

    y_true_file.append(d['y'])
    y_pred_file.append(pred)
    y_prob_file.append(mean_top_prob)

    results.append({
        'filename': d['name'],
        'true_label': d['y'],
        'pred_label': pred,
        'prob': round(mean_top_prob, 4)
    })

# 指标计算
tn, fp, fn, tp = confusion_matrix(y_true_file, y_pred_file).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)
auc = roc_auc_score(y_true_file, y_prob_file)

print("-" * 50)
print(f"🔥 Final Performance (Threshold={BEST_THRESHOLD}, Weight={BEST_WEIGHT}):")
print(f"Sensitivity (SE): {se:.4f}")
print(f"Specificity (SP): {sp:.4f}")
print(f"ICBHI Score: {(se + sp) / 2:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("-" * 50)
print("\nConfusion Matrix:")
print(f"TN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")
print("\nDetailed Classification Report:")
print(classification_report(y_true_file, y_pred_file, target_names=['Normal', 'Abnormal']))

# --- 5. 保存预测结果 CSV ---
df_results = pd.DataFrame(results)
df_results.to_csv("baseline_test_results.csv", index=False)
print(f"📝 详细预测列表已保存至: baseline_test_results.csv")