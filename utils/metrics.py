import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score

def _scan_best(y_true, y_prob, mode="f1_sp", min_sp=0.65, beta=1.0):
    """
    mode:
      - "f1":     只最大化 F1（你现在的）
      - "f1_sp":  在 SP>=min_sp 的约束下最大化 F1；如果达不到约束，就最大化 (F1 + SP)/2
      - "f_beta": 最大化 F_beta（beta>1更偏Recall，beta<1更偏Precision，间接提升SP）
    """
    best = {"score": -1, "f1": 0, "thr": 0.5, "acc": 0, "se": 0, "sp": 0, "cm": None}

    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob > thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)  # recall
        sp = tn / (tn + fp + 1e-8)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        if mode == "f1":
            score = f1
        elif mode == "f1_sp":
            # 先满足SP门槛，再追F1
            if sp >= min_sp:
                score = f1 + 1e-3 * sp  # 轻微打破平局
            else:
                score = 0.5 * (f1 + sp)  # 达不到门槛时折中
        elif mode == "f_beta":
            # F_beta 用 precision/recall 计算，beta<1会更偏precision->通常SP更高
            precision = tp / (tp + fp + 1e-8)
            score = (1 + beta**2) * precision * se / (beta**2 * precision + se + 1e-8)
        else:
            raise ValueError("unknown mode")

        if score > best["score"]:
            best.update({"score": score, "f1": f1, "thr": thr, "acc": acc, "se": se, "sp": sp, "cm": cm})

    return best

def segment_metrics(y_true, y_prob, mode="f1_sp", min_sp=0.65):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    best = _scan_best(y_true, y_prob, mode=mode, min_sp=min_sp)
    return auc, best

def user_metrics(df, probs, mode="f1_sp", min_sp=0.65):
    tmp = df.copy()
    tmp["prob"] = probs
    user_df = tmp.groupby("user_id").agg({"prob": "mean", "label": "max"}).reset_index()
    y_true = user_df["label"].values
    y_prob = user_df["prob"].values
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    best = _scan_best(y_true, y_prob, mode=mode, min_sp=min_sp)
    return auc, best
