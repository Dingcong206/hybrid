import os
import glob
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

# （可选）概率校准：默认不启用
from sklearn.calibration import CalibratedClassifierCV


# =========================
# 1) 路径配置
# =========================
FEAT_DIR = "/data/dingcong/hybrid/hear_features_official"   # 每个文件: (n_seg,512) 或 (512,)
LABEL_DIR = "/data/dingcong/hybrid/audio_and_txt_files"     # ICBHI .txt 标签目录
SAVE_MODEL_NAME = "hear_lr_patientwise_seg_topkmean.joblib"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# =========================
# 2) 阈值扫描约束（为了降低 FN）
# =========================
MIN_SP = 0.45
MIN_SE = 0.55
THR_GRID = np.linspace(0.05, 0.95, 181)

# =========================
# 3) 聚合策略
# =========================
AGG_METHOD = "topk_mean"  # "topk_mean" | "max" | "noisy_or"
TOPK = 3                  # topk_mean 时使用

# =========================
# 4) 模型训练策略
# =========================
USE_CALIBRATION = False   # ⚠️ 默认关掉：校准常导致异常概率更保守 -> FN更高
CALIB_CV = 3              # 只有 USE_CALIBRATION=True 才用

CLASS_WEIGHT = {0: 1.0, 1: 2.5}  # 正类更重 -> 降 FN
LR_C = 1.0
MAX_ITER = 5000


# =========================
# 5) 解析 ICBHI 标签
# =========================
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


def parse_patient_id(base_name: str):
    # ICBHI 文件名常见：101_1b1_Al_sc_Meditron -> patient=101
    if "_" in base_name:
        return base_name.split("_", 1)[0]
    return base_name


# =========================
# 6) 阈值扫描：ICBHI = (SE+SP)/2
#    + 约束 SP >= MIN_SP 且 SE >= MIN_SE
# =========================
def scan_threshold_icbhi(y_true, y_prob, thresholds=THR_GRID, min_sp=MIN_SP, min_se=MIN_SE):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best = None
    best_any = (-1.0, 0.5, 0.0, 0.0, None)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        se = tp / (tp + fn + 1e-9)
        sp = tn / (tn + fp + 1e-9)
        icbhi = 0.5 * (se + sp)

        # 无约束兜底最优
        if icbhi > best_any[0]:
            best_any = (icbhi, float(thr), float(se), float(sp), cm)

        # ✅ 同时约束：SP & SE
        if (sp >= min_sp) and (se >= min_se):
            if (best is None) or (icbhi > best[0]):
                best = (icbhi, float(thr), float(se), float(sp), cm)

    return best if best is not None else best_any


# =========================
# 7) 构建 segment-level 数据
# =========================
def prepare_segment_dataset():
    feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npy")))
    print(f"🔍 正在扫描 {len(feat_files)} 个特征文件...")

    seg_X, seg_y, seg_file, seg_patient = [], [], [], []
    file_label = {}
    file_patient = {}

    used_files = 0

    for f_path in feat_files:
        base_name = os.path.basename(f_path).replace(".npy", "")
        txt_path = os.path.join(LABEL_DIR, base_name + ".txt")
        if not os.path.exists(txt_path):
            continue

        label = get_label_from_icbhi_txt(txt_path)
        if label is None:
            continue

        emb = np.load(f_path)
        if emb.ndim == 1:
            emb = emb[None, :]  # (1,512)

        pid = parse_patient_id(base_name)

        file_label[base_name] = int(label)
        file_patient[base_name] = pid
        used_files += 1

        # segment-level：每个 segment 一个样本（注意：这里仍然是 file-level 标签套给所有 segment）
        for i in range(emb.shape[0]):
            seg_X.append(emb[i].astype(np.float32))
            seg_y.append(int(label))
            seg_file.append(base_name)
            seg_patient.append(pid)

    seg_X = np.asarray(seg_X, dtype=np.float32)
    seg_y = np.asarray(seg_y, dtype=np.int64)

    vals = np.array(list(file_label.values()), dtype=int)
    print(f"✅ 文件数: {used_files}")
    print(f"✅ Segment 总数: {len(seg_X)} | 特征维度: {seg_X.shape[1]}")
    print(f"📊 文件级类别分布：正常={(vals==0).sum()}，异常={(vals==1).sum()}")

    return seg_X, seg_y, np.array(seg_file), np.array(seg_patient), file_label, file_patient


# =========================
# 8) patient-wise split（按 patient 划分文件）
# =========================
def patient_wise_split_files(file_label, file_patient, test_size=TEST_SIZE, seed=RANDOM_SEED):
    df = pd.DataFrame({
        "file": list(file_label.keys()),
        "patient": [file_patient[f] for f in file_label.keys()],
        "label": [file_label[f] for f in file_label.keys()]
    })

    # patient 标签：该 patient 任一文件异常 -> patient 异常
    p = df.groupby("patient")["label"].max().reset_index()
    patients = p["patient"].values
    p_labels = p["label"].values.astype(int)

    train_p, val_p = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=p_labels
    )

    train_files = df[df["patient"].isin(train_p)]["file"].tolist()
    val_files = df[df["patient"].isin(val_p)]["file"].tolist()

    print(f"Train patients: {len(train_p)} | Val patients: {len(val_p)}")
    print(f"Train files   : {len(train_files)} | Val files   : {len(val_files)}")

    return set(train_files), set(val_files), set(train_p), set(val_p)


# =========================
# 9) Segment -> File 概率聚合
# =========================
def aggregate_file_max(files, seg_files, seg_probs):
    from collections import defaultdict
    bucket = defaultdict(list)
    for f, p in zip(seg_files, seg_probs):
        if f in files:
            bucket[f].append(float(p))

    out_files, out_probs = [], []
    for f in sorted(files):
        ps = bucket.get(f, [])
        out_files.append(f)
        out_probs.append(float(np.max(ps)) if len(ps) else 0.0)
    return out_files, np.asarray(out_probs, dtype=np.float32)


def aggregate_file_topk_mean(files, seg_files, seg_probs, k=TOPK):
    from collections import defaultdict
    bucket = defaultdict(list)
    for f, p in zip(seg_files, seg_probs):
        if f in files:
            bucket[f].append(float(p))

    out_files, out_probs = [], []
    for f in sorted(files):
        ps = bucket.get(f, [])
        if len(ps) == 0:
            out_files.append(f)
            out_probs.append(0.0)
            continue
        ps = sorted(ps, reverse=True)
        kk = min(k, len(ps))
        out_files.append(f)
        out_probs.append(float(np.mean(ps[:kk])))
    return out_files, np.asarray(out_probs, dtype=np.float32)


def aggregate_file_noisy_or(files, seg_files, seg_probs):
    from collections import defaultdict
    bucket = defaultdict(list)
    for f, p in zip(seg_files, seg_probs):
        if f in files:
            bucket[f].append(float(p))

    out_files, out_probs = [], []
    for f in sorted(files):
        ps = bucket.get(f, [])
        if len(ps) == 0:
            out_files.append(f)
            out_probs.append(0.0)
            continue
        ps = np.clip(np.asarray(ps, dtype=np.float64), 1e-6, 1 - 1e-6)
        p_file = 1.0 - np.prod(1.0 - ps)  # Noisy-OR
        out_files.append(f)
        out_probs.append(float(p_file))
    return out_files, np.asarray(out_probs, dtype=np.float32)


def aggregate_files(files, seg_files, seg_probs):
    if AGG_METHOD == "max":
        return aggregate_file_max(files, seg_files, seg_probs)
    if AGG_METHOD == "noisy_or":
        return aggregate_file_noisy_or(files, seg_files, seg_probs)
    # 默认 topk_mean
    return aggregate_file_topk_mean(files, seg_files, seg_probs, k=TOPK)


# =========================
# 10) 主流程
# =========================
def run():
    seg_X, seg_y, seg_files, seg_patients, file_label, file_patient = prepare_segment_dataset()
    train_files, val_files, train_p, val_p = patient_wise_split_files(file_label, file_patient)

    train_mask = np.isin(seg_files, list(train_files))
    val_mask = np.isin(seg_files, list(val_files))

    X_train, y_train = seg_X[train_mask], seg_y[train_mask]
    X_val_seg, y_val_seg = seg_X[val_mask], seg_y[val_mask]
    val_seg_files = seg_files[val_mask]

    # 标准化（只在训练 segment 上 fit）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_seg_s = scaler.transform(X_val_seg)

    print(f"🚀 Training Logistic Regression (segment-level)...")
    print(f"   class_weight={CLASS_WEIGHT}, C={LR_C}, max_iter={MAX_ITER}, agg={AGG_METHOD}, topk={TOPK}, calibrate={USE_CALIBRATION}")

    base_lr = LogisticRegression(
        max_iter=MAX_ITER,
        C=LR_C,
        solver="lbfgs",
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED
    )

    if USE_CALIBRATION:
        clf = CalibratedClassifierCV(base_lr, method="sigmoid", cv=CALIB_CV)
        clf.fit(X_train_s, y_train)
    else:
        clf = base_lr
        clf.fit(X_train_s, y_train)

    # segment 概率
    val_seg_prob = clf.predict_proba(X_val_seg_s)[:, 1]

    # file-level 聚合
    val_file_list, val_file_prob = aggregate_files(val_files, val_seg_files, val_seg_prob)
    y_val_file = np.array([file_label[f] for f in val_file_list], dtype=int)

    # AUC（file-level）
    auc = roc_auc_score(y_val_file, val_file_prob)

    # 阈值扫描（file-level）——关键：同时约束 SE/SP
    best_icbhi, best_thr, se, sp, cm = scan_threshold_icbhi(
        y_val_file, val_file_prob, thresholds=THR_GRID, min_sp=MIN_SP, min_se=MIN_SE
    )

    print("\n" + "=" * 70)
    print(f"🎯 Val FILE-level AUC-ROC: {auc:.4f}")
    print(f"⭐ Best ICBHI: {best_icbhi:.4f} | SE: {se:.4f} | SP: {sp:.4f} | thr: {best_thr:.2f} | min_sp={MIN_SP} | min_se={MIN_SE}")
    print("🧱 Confusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)
    print("=" * 70)

    # 保存
    import joblib
    joblib.dump({"model": clf, "scaler": scaler}, SAVE_MODEL_NAME)
    print(f"💾 Saved: {SAVE_MODEL_NAME}")


if __name__ == "__main__":
    run()
