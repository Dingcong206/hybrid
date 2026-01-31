import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- 1. 路径配置 ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official_final"
# 假设你的标签文件是一个 CSV，包含 [file_name, label]
LABEL_CSV = "/data/dingcong/hybrid/labels.csv"


# --- 2. 加载与聚合特征 ---
def load_and_pool_features(feat_dir, label_df):
    X = []
    y = []

    print("🔄 正在聚合音频特征 (Mean Pooling)...")
    for _, row in label_df.iterrows():
        feat_path = os.path.join(feat_dir, row['file_name'].replace('.wav', '.npy'))

        if os.path.exists(feat_path):
            # 加载特征，形状通常是 (N_segments, 512)
            feat = np.load(feat_path)

            # Mean Pooling: 将 N 个片段压扁成 1 个 512 维向量
            # 这样每个音频就只有一个特征表示
            pooled_feat = np.mean(feat, axis=0)

            X.append(pooled_feat)
            y.append(row['label'])

    return np.array(X), np.array(y)


# --- 3. 准备数据 ---
# 假设你的 label_df 已经准备好
# df = pd.read_csv(LABEL_CSV)
# X, y = load_and_pool_features(FEAT_DIR, df)

# 如果你还没准备好 CSV，这里写一个模拟逻辑
# X = np.random.randn(920, 512) # 模拟 920 个 512 维特征
# y = np.random.randint(0, 2, 920) # 模拟 0/1 标签

# --- 4. 训练与评估 ---
# A. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# B. 标准化 (对于 LR 非常重要)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# C. 训练逻辑回归
# 官方 notebook 推荐使用 L2 正则化以防止过拟合
print("🚀 正在训练 Logistic Regression 分类器...")
clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
clf.fit(X_train, y_train)

# D. 评估
y_pred = clf.predict(X_test)
print("\n📊 分类报告:")
print(classification_report(y_test, y_pred))

# E. 混淆矩阵
print("🧱 混淆矩阵:")
print(confusion_matrix(y_test, y_pred))