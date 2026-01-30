import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score


def _scan_best_f1(y_true, y_prob):
    best = {
        "f1": 0,
        "thr": 0.5,
        "acc": 0,
        "se": 0,
        "sp": 0,
        "cm": None
    }

    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        if f1 > best["f1"]:
            best.update({
                "f1": f1,
                "thr": thr,
                "acc": acc,
                "se": se,
                "sp": sp,
                "cm": cm
            })

    return best


def segment_metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    best = _scan_best_f1(y_true, y_prob)
    return auc, best


def user_metrics(df, probs):
    """
    df: 包含 user_id, label
    probs: 每个 sample 的预测概率
    """
    tmp = df.copy()
    tmp["prob"] = probs

    user_df = tmp.groupby("user_id").agg({
        "prob": "mean",
        "label": "max"
    }).reset_index()

    y_true = user_df["label"].values
    y_prob = user_df["prob"].values

    auc = roc_auc_score(y_true, y_prob)
    best = _scan_best_f1(y_true, y_prob)
    return auc, best
