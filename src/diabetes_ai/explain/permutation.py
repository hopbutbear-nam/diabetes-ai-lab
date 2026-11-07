"""Độ quan trọng hoán vị toàn cục (Permutation Feature Importance)."""

from typing import Any, Callable
import numpy as np

from diabetes_ai.evaluation.metrics import macro_f1, roc_auc_ovr
from diabetes_ai.features.feature_names import vi


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    n_repeats: int = 5,
    seed: int = 42,
    metric: str = "roc_auc_ovr",
) -> list[dict[str, Any]]:
    """Tính độ quan trọng hoán vị của từng đặc trưng trên tập dữ liệu.

    Args:
        predict_fn: Hàm nhận ma trận đặc trưng và trả về xác suất dự đoán (n_samples, n_classes).
        X: Ma trận đặc trưng đánh giá (thường là validation set).
        y: Mảng nhãn thực tế.
        features: Danh sách tên đặc trưng tương ứng với các cột của X.
        n_repeats: Số lần xáo trộn ngẫu nhiên độc lập cho mỗi cột. Mặc định là 5.
        seed: Seed sinh số ngẫu nhiên đảm bảo tính tái lập. Mặc định là 42.
        metric: Thước đo đánh giá drop, chấp nhận "roc_auc_ovr" hoặc "macro_f1".

    Returns:
        list[dict[str, Any]]: Danh sách kết quả cho từng đặc trưng gồm:
            "feature", "vi", "mean_drop", "std_drop", sắp xếp giảm dần theo "mean_drop".
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)

    if len(features) != X_arr.shape[1]:
        raise ValueError(
            f"Số lượng đặc trưng ({len(features)}) không khớp số cột của X ({X_arr.shape[1]})."
        )

    base_proba = predict_fn(X_arr)
    base_score = _calculate_metric(base_proba, y_arr, metric)

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    # Sao chép một lần để hoán vị từng cột tại chỗ và hoàn tác
    X_perm = X_arr.copy()

    for j, feat in enumerate(features):
        orig_col = X_perm[:, j].copy()
        drops: list[float] = []

        for _ in range(n_repeats):
            X_perm[:, j] = rng.permutation(orig_col)
            perm_proba = predict_fn(X_perm)
            perm_score = _calculate_metric(perm_proba, y_arr, metric)
            drop = base_score - perm_score
            drops.append(float(drop))

        X_perm[:, j] = orig_col  # Khôi phục giá trị cột gốc

        mean_drop = float(np.mean(drops))
        std_drop = float(np.std(drops))

        results.append({
            "feature": feat,
            "vi": vi(feat),
            "mean_drop": mean_drop,
            "std_drop": std_drop,
        })

    results.sort(key=lambda r: r["mean_drop"], reverse=True)
    return results


def _calculate_metric(proba: np.ndarray, y: np.ndarray, metric: str) -> float:
    """Tính điểm theo thước đo chỉ định."""
    proba_arr = np.asarray(proba, dtype=float)
    if metric == "roc_auc_ovr":
        return roc_auc_ovr(y, proba_arr)
    elif metric == "macro_f1":
        if proba_arr.ndim > 1 and proba_arr.shape[1] > 1:
            y_pred = np.argmax(proba_arr, axis=1)
            n_classes = proba_arr.shape[1]
        else:
            y_pred = np.round(proba_arr.flatten()).astype(int)
            n_classes = len(np.unique(y))
        return macro_f1(y, y_pred, n_classes=n_classes)
    else:
        raise ValueError(
            f"Không hỗ trợ thước đo đánh giá: '{metric}'. Chỉ chấp nhận 'roc_auc_ovr' hoặc 'macro_f1'."
        )
