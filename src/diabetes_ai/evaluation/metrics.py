"""ROC-AUC OvR, PR-AUC, macro-F1, recall/precision từng lớp, Brier, ECE."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def roc_auc_per_class(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Tính ROC-AUC OvR cho từng lớp.

    Args:
        y_true: Mảng nhãn số nguyên (n,).
        proba: Ma trận xác suất (n, K).

    Returns:
        np.ndarray: AUC từng lớp; lớp không có mẫu dương → nan.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    K = proba.shape[1]
    aucs = np.full(K, np.nan)
    for k in range(K):
        y_bin = (y_true == k).astype(np.int32)
        if y_bin.sum() == 0:
            continue
        try:
            aucs[k] = roc_auc_score(y_bin, proba[:, k])
        except Exception:
            aucs[k] = np.nan
    return aucs


def roc_auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Tính ROC-AUC macro OvR (trung bình các lớp có mẫu dương).

    Khi K=2, bằng AUC của cột 1.

    Args:
        y_true: Mảng nhãn số nguyên (n,).
        proba: Ma trận xác suất (n, K).

    Returns:
        float: Giá trị AUC macro.
    """
    per_class = roc_auc_per_class(y_true, proba)
    valid = per_class[~np.isnan(per_class)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


# ---------------------------------------------------------------------------
# PR-AUC helpers
# ---------------------------------------------------------------------------

def pr_auc_per_class(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Tính PR-AUC OvR cho từng lớp (average_precision_score).

    Args:
        y_true: Mảng nhãn số nguyên (n,).
        proba: Ma trận xác suất (n, K).

    Returns:
        np.ndarray: PR-AUC từng lớp; lớp không có mẫu dương → nan.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    K = proba.shape[1]
    aucs = np.full(K, np.nan)
    for k in range(K):
        y_bin = (y_true == k).astype(np.int32)
        if y_bin.sum() == 0:
            continue
        try:
            aucs[k] = average_precision_score(y_bin, proba[:, k])
        except Exception:
            aucs[k] = np.nan
    return aucs


def pr_auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Tính PR-AUC macro OvR (trung bình bỏ nan).

    Args:
        y_true: Mảng nhãn số nguyên (n,).
        proba: Ma trận xác suất (n, K).

    Returns:
        float: Giá trị PR-AUC macro.
    """
    per_class = pr_auc_per_class(y_true, proba)
    valid = per_class[~np.isnan(per_class)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


# ---------------------------------------------------------------------------
# F1, Recall, Precision
# ---------------------------------------------------------------------------

def recall_precision_f1_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, list[float]]:
    """Tính recall, precision, f1 cho từng lớp.

    Chia 0 → 0.0.

    Args:
        y_true: Mảng nhãn thật (n,).
        y_pred: Mảng nhãn dự đoán (n,).
        n_classes: Số lượng lớp.

    Returns:
        dict với khóa 'recall', 'precision', 'f1', mỗi khóa là list độ dài n_classes.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recall_list: list[float] = []
    precision_list: list[float] = []
    f1_list: list[float] = []
    for k in range(n_classes):
        tp = int(np.sum((y_true == k) & (y_pred == k)))
        fn = int(np.sum((y_true == k) & (y_pred != k)))
        fp = int(np.sum((y_true != k) & (y_pred == k)))
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        denom = prec + rec
        f1 = 2 * prec * rec / denom if denom > 0 else 0.0
        recall_list.append(float(rec))
        precision_list.append(float(prec))
        f1_list.append(float(f1))
    return {"recall": recall_list, "precision": precision_list, "f1": f1_list}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Tính macro F1 trung bình các lớp.

    Args:
        y_true: Mảng nhãn thật (n,).
        y_pred: Mảng nhãn dự đoán (n,).
        n_classes: Số lượng lớp.

    Returns:
        float: Macro F1.
    """
    rpf = recall_precision_f1_per_class(y_true, y_pred, n_classes)
    return float(np.mean(rpf["f1"]))


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Tính Brier score đa lớp.

    = trung bình theo mẫu của Σ_k (p_ik − 1[y_i=k])²

    Args:
        y_true: Mảng nhãn số nguyên (n,).
        proba: Ma trận xác suất (n, K).

    Returns:
        float: Giá trị Brier score.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    n, K = proba.shape
    # one-hot
    one_hot = np.zeros((n, K), dtype=np.float64)
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def brier_per_class(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Tính Brier score nhị phân OvR cho từng lớp.

    Args:
        y_true: Mảng nhãn số nguyên (n,).
        proba: Ma trận xác suất (n, K).

    Returns:
        np.ndarray: Brier score nhị phân từng lớp.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    K = proba.shape[1]
    scores = np.zeros(K, dtype=np.float64)
    for k in range(K):
        y_bin = (y_true == k).astype(np.float64)
        scores[k] = float(np.mean((proba[:, k] - y_bin) ** 2))
    return scores


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def ece_binary(
    y_bin: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Tính ECE nhị phân.

    Bin đều np.linspace(0,1,n_bins+1); gán bằng np.digitize(p, edges[1:-1]);
    ECE = Σ (n_b/N)·|mean(y_b) − mean(p_b)|.

    Args:
        y_bin: Nhãn nhị phân (n,).
        p: Xác suất dự đoán (n,).
        n_bins: Số bin (mặc định 10).

    Returns:
        float: ECE.
    """
    y_bin = np.asarray(y_bin, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    N = len(p)
    if N == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1])  # 0 .. n_bins-1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(np.mean(y_bin[mask]))
        conf_b = float(np.mean(p[mask]))
        ece += (n_b / N) * abs(acc_b - conf_b)
    return float(ece)


def ece_ovr(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    """Tính ECE OvR cho từng lớp và macro.

    Args:
        y_true: Mảng nhãn số nguyên (n,).
        proba: Ma trận xác suất (n, K).
        n_bins: Số bin (mặc định 10).

    Returns:
        dict với khóa 'per_class' (list float) và 'macro' (float).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    K = proba.shape[1]
    per_class = []
    for k in range(K):
        y_bin = (y_true == k).astype(np.float64)
        per_class.append(ece_binary(y_bin, proba[:, k], n_bins=n_bins))
    return {"per_class": per_class, "macro": float(np.mean(per_class))}


# ---------------------------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------------------------

def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    """Tính độ chính xác baseline (luôn đoán lớp đa số).

    Args:
        y_true: Mảng nhãn số nguyên (n,).

    Returns:
        float: Tỉ lệ mẫu thuộc lớp đa số.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(y_true) == 0:
        return 0.0
    counts = np.bincount(y_true)
    return float(counts.max() / len(y_true))


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def compute_all(
    y_true: np.ndarray,
    proba: np.ndarray,
    y_pred: np.ndarray | None = None,
    class_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict:
    """Tính tất cả chỉ số đánh giá và trả về dict có thể ghi JSON.

    Args:
        y_true: Mảng nhãn thật int64 (n,).
        proba: Ma trận xác suất (n, K).
        y_pred: Nhãn dự đoán; nếu None sẽ lấy argmax(proba).
        class_names: Tên các lớp.
        n_bins: Số bin cho ECE.

    Returns:
        dict với các khóa: n, accuracy, accuracy_majority_baseline,
        roc_auc_ovr_macro, roc_auc_per_class, pr_auc_macro, pr_auc_per_class,
        macro_f1, recall_per_class, precision_per_class, f1_per_class,
        brier_multiclass, brier_per_class, ece_per_class, ece_macro,
        confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    n, K = proba.shape

    if y_pred is None:
        y_pred_arr = np.argmax(proba, axis=1).astype(np.int64)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    rpf = recall_precision_f1_per_class(y_true, y_pred_arr, K)
    ece_res = ece_ovr(y_true, proba, n_bins=n_bins)

    # Confusion matrix: list of lists
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred_arr):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1
    cm_list = cm.tolist()

    return {
        "n": int(n),
        "accuracy": float(np.mean(y_true == y_pred_arr)),
        "accuracy_majority_baseline": majority_baseline_accuracy(y_true),
        "roc_auc_ovr_macro": roc_auc_ovr(y_true, proba),
        "roc_auc_per_class": roc_auc_per_class(y_true, proba).tolist(),
        "pr_auc_macro": pr_auc_ovr(y_true, proba),
        "pr_auc_per_class": pr_auc_per_class(y_true, proba).tolist(),
        "macro_f1": macro_f1(y_true, y_pred_arr, K),
        "recall_per_class": rpf["recall"],
        "precision_per_class": rpf["precision"],
        "f1_per_class": rpf["f1"],
        "brier_multiclass": brier_multiclass(y_true, proba),
        "brier_per_class": brier_per_class(y_true, proba).tolist(),
        "ece_per_class": ece_res["per_class"],
        "ece_macro": ece_res["macro"],
        "confusion_matrix": cm_list,
    }
