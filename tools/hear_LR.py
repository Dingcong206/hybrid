import os
import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- 1. 路径配置 ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official"  # .npy 特征目录
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"  # ICBHI .txt 标签目录
SAVE_MODEL_NAME = "hear_lr_model.joblib"


# --- 2. 标签解析函数 (针对 ICBHI 标准 TXT 格式) ---
def get_label_from_icbhi_txt(txt_path):
    """
    解析 ICBHI TXT 文件。
    逻辑：只要该音频中任何一个时间段出现了 湿啰音(Crackles) 或 哮鸣音(Wheezes)，
    我们就将其标记为异常 (1)，否则为正常 (0)。
    """
    try:
        # ICBHI 格式：[start_time, end_time, crackle, wheeze]
        # 有些文件用制表符 \t，有些用空格，sep=None 可以自动处理
        df = pd.read_csv(txt_path, sep=None, engine='python', header=None)

        # 只要第2列或第3列中有 1，就判定为正样本 (1)
        has_crackles = (df[2] == 1).any()
        has_wheezes = (df[3] == 1).any()

        return 1 if (has_crackles or has_wheezes) else 0
    except Exception as e:
        print(f"⚠️ 解析标签失败 {txt_path}: {e}")
        return None


# --- 3. 数据集构建逻辑 ---
def prepare_dataset():
    X, y = [], []
    feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
    print(f"🔍 正在处理 {len(feat_files)} 个特征文件...")

    for f_path in feat_files:
        base_name = os.path.basename(f_path).replace('.npy', '')
        txt_path = os.path.join(LABEL_DIR, base_name + ".txt")

        if not os.path.exists(txt_path):
            continue

        # A. 获取标签
        label = get_label_from_icbhi_txt(txt_path)
        if label is None: continue

        # B. 加载特征并池化
        # HeAR 提取出的特征形状通常是 (N_segments, 512)
        # 我们对 N 个片段取平均值，得到代表全长的 512 维特征
        embeddings = np.load(f_path)
        if embeddings.ndim > 1:
            pooled_feat = np.mean(embeddings, axis=0)
        else:
            pooled_feat = embeddings

        X.append(pooled_feat)
        y.append(label)

    return np.array(X), np.array(y)


# --- 4. 训练与评估流水线 ---
def run_training():
    # A. 准备数据
    X, y = prepare_dataset()
    print(f"✅ 数据准备就绪：样本数={len(X)}, 特征维度={X.shape[1]}")
    print(f"📊 类别分布：正常={sum(y == 0)}, 异常={sum(y == 1)}")

    # B. 划分数据集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # C. 特征标准化 (LR 对量纲非常敏感)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # D. 训练逻辑回归
    # 使用 class_weight='balanced' 自动处理类别不平衡
    # 增加 max_iter 确保在大模型特征下收敛
    print("🚀 开始训练 Logistic Regression 模型...")
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # E. 模型评估
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "=" * 30)
    print("📋 分类评估报告:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    print(f"🎯 AUC-ROC 分数: {roc_auc_score(y_test, y_prob):.4f}")

    print("\n🧱 混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 30)

    # F. (可选) 保存模型
    import joblib
    joblib.dump({'model': clf, 'scaler': scaler}, SAVE_MODEL_NAME)
    print(f"💾 模型已保存至: {SAVE_MODEL_NAME}")


if __name__ == "__main__":
    run_training()