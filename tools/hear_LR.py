import os
import glob
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

# --- 1. 路径配置 ---
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official"   # .npy 特征目录 (每个文件 shape: [n_seg,512] 或 [512])
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"     # ICBHI .txt 标签目录
SAVE_MODEL_NAME = "hear_lr_patientwise.joblib"

RANDOM_SEED = 42
TEST_SIZE = 0.2


# --- 2. 标签解析函数 (针对 ICBHI 标准 TXT 格式) ---
def get_label_from_icbhi_txt(txt_path):
    """
    ICBHI TXT: [start_time, end_time, crackle, wheeze]
    只要任一段出现 crackle 或 wheeze -> 异常(1)，否则正常(0)
    """
    try:
        df = pd.read_csv(txt_path, sep=None, engine="python", header=None)
        has_crackles = (df[2] == 1).any()
        has_wheezes = (df[3] == 1).any()
        return 1 if (has_crackles or has_wheezes) else 0
    except Exception as e:
        print(f"⚠️ 解析标签失败 {txt_path}: {e}")
        return None


# --- 3. 解析 patient_id：ICBHI 文件名通常形如 101_1b1_Al_sc_Meditron ---
def parse_patient_id(base_name: str):
    """
    从 base_name 提取 patient_id。默认取第一个 '_' 之前的字段。
    如果不是纯数字，也照样当作字符串 patient_id 用。
    """
    if "_" in base_name:
        return base_name.split("_", 1)[0]
    return base_name  # 兜底


# --- 4. 阈值扫描：按 ICBHI score 选最佳阈值 ---
def scan_threshold_icbhi(y_true, y_prob, thresholds=None):
    """
    y_true: {0,1}
    y_prob: [0,1]
    返回 best: (best_icbhi, best_thr, se, sp, cm)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_icbhi, best_thr, best_se, best_sp, best_cm = -1.0, 0.5, 0.0, 0.0, None

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        se = tp / (tp + fn + 1e-9)  # sensitivity/recall for class 1
        sp = tn / (tn + fp + 1e-9)  # specificity for class 0
        icbhi = 0.5 * (se + sp)

        if icbhi > best_icbhi:
            best_icbhi, best_thr, best_se, best_sp, best_cm = icbhi, float(thr), float(se), float(sp), cm

    return best_icbhi, best_thr, best_se, best_sp, best_cm


# --- 5. 数据集构建：文件级 pooled embedding + 标签 + patient_id ---
def prepare_dataset():
    rows = []
    feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
    print(f"🔍 正在处理 {len(feat_files)} 个特征文件...")

    for f_path in feat_files:
        base_name = os.path.basename(f_path).replace(".npy", "")
        txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
        if not os.path.exists(txt_path):
            continue

        label = get_label_from_icbhi_txt(txt_path)
        if label is None:
            continue

        emb = np.load(f_path)
        if emb.ndim > 1:
            feat = emb.mean(axis=0)  # (512,)
        else:
            feat = emb

        patient_id = parse_patient_id(base_name)
        rows.append((base_name, patient_id, label, feat))

    if not rows:
        raise RuntimeError("没有匹配到任何 (npy, txt) 对，请检查 FEAT_DIR / LABEL_DIR 路径。")

    # 打包
    file_names = [r[0] for r in rows]
    patient_ids = [r[1] for r in rows]
    labels = np.array([r[2] for r in rows], dtype=int)
    X = np.stack([r[3] for r in rows], axis=0).astype(np.float32)

    print(f"✅ 数据准备就绪：样本数={len(X)}, 特征维度={X.shape[1]}")
    print(f"📊 文件级类别分布：正常={int((labels==0).sum())}, 异常={int((labels==1).sum())}")

    return file_names, patient_ids, X, labels


# --- 6. patient-wise split：按 patient 分 train/val ---
def patient_wise_split(file_names, patient_ids, labels, test_size=0.2, seed=42):
    df = pd.DataFrame({
        "file_name": file_names,
        "patient_id": patient_ids,
        "label": labels
    })

    # patient 标签：只要该 patient 任何文件异常 -> patient 异常（更接近临床）
    p = df.groupby("patient_id")["label"].max().reset_index()
    patients = p["patient_id"].values
    p_labels = p["label"].values.astype(int)

    train_p, val_p = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=p_labels
    )

    train_mask = df["patient_id"].isin(train_p).values
    val_mask = df["patient_id"].isin(val_p).values

    return df, train_mask, val_mask, train_p, val_p


# --- 7. 主流程 ---
def run_training():
    file_names, patient_ids, X, y = prepare_dataset()

    df_meta, train_mask, val_mask, train_p, val_p = patient_wise_split(
        file_names, patient_ids, y, test_size=TEST_SIZE, seed=RANDOM_SEED
    )

    print(f"Train patients: {len(train_p)} | Val patients: {len(val_p)}")
    print(f"Train files   : {int(train_mask.sum())} | Val files   : {int(val_mask.sum())}")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # LR：balanced
    print("🚀 开始训练 Logistic Regression (class_weight=balanced, patient-wise split)...")
    clf = LogisticRegression(
        max_iter=5000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_SEED
    )
    clf.fit(X_train_s, y_train)

    # 预测概率
    val_prob = clf.predict_proba(X_val_s)[:, 1]
    auc = roc_auc_score(y_val, val_prob)

    # 阈值扫描（ICBHI）
    best_icbhi, best_thr, se, sp, cm = scan_threshold_icbhi(y_val, val_prob)

    print("\n" + "=" * 60)
    print(f"🎯 Val AUC-ROC: {auc:.4f}")
    print(f"⭐ Best ICBHI: {best_icbhi:.4f} | SE: {se:.4f} | SP: {sp:.4f} | thr: {best_thr:.2f}")
    print("🧱 Confusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)
    print("=" * 60)

    # 保存模型（含 scaler）
    import joblib
    joblib.dump({"model": clf, "scaler": scaler}, SAVE_MODEL_NAME)
    print(f"💾 模型已保存至: {SAVE_MODEL_NAME}")


if __name__ == "__main__":
    run_training()
