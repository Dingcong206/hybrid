import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- 1. 路径配置 ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official_baseline"
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"


# --- 2. 标签解析函数 (ICBHI 逻辑) ---
def get_label(base_name):
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
    if not os.path.exists(txt_path): return None
    df = pd.read_csv(txt_path, sep='\t', header=None)  # ICBHI 默认制表符
    # 只要有 crackle(col 2) 或 wheeze(col 3) 就是异常(1)
    return 1 if (df[2] == 1).any() or (df[3] == 1).any() else 0


# --- 3. 数据加载 ---
X, y = [], []
print("🚚 正在加载特征...")
feat_files = glob.glob(os.path.join(FEAT_DIR, "*.npy"))

for f_path in feat_files:
    base_name = os.path.basename(f_path).replace(".npy", "")
    label = get_label(base_name)

    if label is not None:
        feat = np.load(f_path).squeeze()  # 确保是 (512,)
        X.append(feat)
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f"✅ 加载完成: 样本数={len(X)}, 特征维度={X.shape[1]}")

# --- 4. 训练与评估 ---
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 标准化 (非常重要，LR 对量纲敏感)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化 LR
# class_weight='balanced' 能有效平衡你的正常/异常比例，降低 FN
#model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1.0, 1: 2},  # 正常给1，异常给2
    C=1.0,
    random_state=42
)
model.fit(X_train, y_train)

# --- 5. 结果输出 ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 30 + " 结果报告 " + "=" * 30)
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("\n分类详情:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))