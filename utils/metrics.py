import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score


def _scan_best(y_true, y_prob, mode="f1_sp", min_sp=0.65, beta=1.0):
    best = {"score": -1, "f1": 0, "thr": 0.5, "acc": 0, "se": 0, "sp": 0, "cm": None}

    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob > thr).astype(int)

        # 确保预测包含两类，防止混淆矩阵维度不对
        if len(np.unique(y_true)) < 2:
            continue

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        # 指标选择逻辑
        if mode == "f1":
            score = f1
        elif mode == "f1_sp":
            if sp >= min_sp:
                score = f1 + 1e-3 * sp
            else:
                score = 0.5 * (f1 + sp)
        else:
            score = f1

        if score > best["score"]:
            best.update({
                "score": score, "f1": f1, "thr": thr,
                "acc": acc, "se": se, "sp": sp, "cm": cm  # ✅ 包含混淆矩阵
            })
    return best


def segment_metrics(y_true, y_prob, mode="f1_sp", min_sp=0.65):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    best = _scan_best(y_true, y_prob, mode=mode, min_sp=min_sp)
    return auc, best


def user_metrics(df, probs, mode="f1_sp", min_sp=0.65):
    tmp = df.copy()
    tmp["prob"] = probs
    # 聚合逻辑：你可以试着将 mean 改为 max 看看 F1 是否会更高
    user_df = tmp.groupby("user_id").agg({"prob": "mean", "label": "max"}).reset_index()

    y_true = user_df["label"].values
    y_prob = user_df["prob"].values
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    best = _scan_best(y_true, y_prob, mode=mode, min_sp=min_sp)
    return auc, best