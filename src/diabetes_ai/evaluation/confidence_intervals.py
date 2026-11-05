"""Tính toán khoảng tin cậy Wilson và Bootstrap phân tầng cho các chỉ số đánh giá."""

from __future__ import annotations

from typing import Callable

import numpy as np

from diabetes_ai.evaluation.metrics import recall_precision_f1_per_class, roc_auc_ovr


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Tính khoảng tin cậy Wilson score cho tỷ lệ nhị phân k/n.

    Args:
        k: Số lần thành công (k >= 0).
        n: Tổng số lần thử (n >= 0).
        z: Phân vị chuẩn (mặc định 1.959964 ứng với mức tin cậy 95%).

    Returns:
        tuple[float, float]: (low, high) giới hạn trong khoảng [0.0, 1.0].
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tạo chỉ số lấy mẫu bootstrap có hoàn lại theo từng lớp (stratified).

    Bảo toàn số lượng mẫu chính xác của từng lớp như trong mảng y ban đầu.

    Args:
        y: Mảng nhãn đầu vào (n,).
        rng: Bộ sinh số ngẫu nhiên numpy Generator.

    Returns:
        np.ndarray: Mảng chỉ số resample (n,) kiểu int64.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)
    if n == 0:
        return np.array([], dtype=np.int64)

    classes = np.unique(y_arr)
    sampled_indices: list[np.ndarray] = []
    for c in classes:
        idx_c = np.flatnonzero(y_arr == c)
        if len(idx_c) > 0:
            sampled_c = rng.choice(idx_c, size=len(idx_c), replace=True)
            sampled_indices.append(sampled_c)

    boot_indices = np.concatenate(sampled_indices).astype(np.int64)
    rng.shuffle(boot_indices)
    return boot_indices


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    stratified: bool = True,
) -> dict[str, float | int]:
    """Ước lượng khoảng tin cậy bootstrap cho hàm metric_fn(y_true, proba).

    Args:
        y_true: Mảng nhãn thực tế (n,).
        proba: Mảng xác suất hoặc dự đoán tương thích với metric_fn.
        metric_fn: Hàm nhận (y, proba) trả về điểm số float.
        n: Số lượt lấy mẫu bootstrap (mặc định 1000).
        ci: Mức độ tin cậy (mặc định 0.95).
        seed: Seed sinh số ngẫu nhiên (mặc định 42).
        stratified: Có phân tầng theo y_true hay không (mặc định True).

    Returns:
        dict với các khóa: point, low, high, n_boot.
    """
    y_arr = np.asarray(y_true)
    proba_arr = np.asarray(proba)
    point = float(metric_fn(y_arr, proba_arr))

    rng = np.random.default_rng(seed)
    boot_scores: list[float] = []
    N = len(y_arr)

    for _ in range(n):
        if stratified:
            idx = stratified_bootstrap_indices(y_arr, rng)
        else:
            idx = rng.choice(N, size=N, replace=True)
        score = float(metric_fn(y_arr[idx], proba_arr[idx]))
        boot_scores.append(score)

    alpha = 1.0 - ci
    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0

    low = float(np.nanpercentile(boot_scores, low_p))
    high = float(np.nanpercentile(boot_scores, high_p))

    return {
        "point": point,
        "low": low,
        "high": high,
        "n_boot": n,
    }


def bootstrap_multi(
    y_true: np.ndarray,
    proba: np.ndarray,
    y_pred: np.ndarray,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    """Tính điểm và khoảng tin cậy bootstrap cho roc_auc_ovr_macro và recall_per_class.

    Dùng chung chỉ số bootstrap trong một vòng lặp để tối ưu hóa hiệu năng tính toán.

    Args:
        y_true: Mảng nhãn thực tế (n,).
        proba: Ma trận xác suất dự đoán (n, K).
        y_pred: Mảng nhãn dự đoán (n,).
        n: Số lượt lấy mẫu bootstrap (mặc định 1000).
        ci: Mức độ tin cậy (mặc định 0.95).
        seed: Seed sinh số ngẫu nhiên (mặc định 42).

    Returns:
        dict chứa kết quả cho 'roc_auc_ovr_macro' và 'recall_per_class'.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    proba_arr = np.asarray(proba, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    n_classes = proba_arr.shape[1] if proba_arr.ndim == 2 else int(max(np.max(y_true_arr), np.max(y_pred_arr))) + 1

    point_auc = roc_auc_ovr(y_true_arr, proba_arr)
    point_rec = recall_precision_f1_per_class(y_true_arr, y_pred_arr, n_classes)["recall"]

    rng = np.random.default_rng(seed)
    boot_aucs: list[float] = []
    boot_recalls: list[list[float]] = [[] for _ in range(n_classes)]

    for _ in range(n):
        idx = stratified_bootstrap_indices(y_true_arr, rng)
        b_auc = roc_auc_ovr(y_true_arr[idx], proba_arr[idx])
        boot_aucs.append(b_auc)

        b_rec = recall_precision_f1_per_class(y_true_arr[idx], y_pred_arr[idx], n_classes)["recall"]
        for c in range(n_classes):
            boot_recalls[c].append(b_rec[c])

    alpha = 1.0 - ci
    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0

    auc_low = float(np.nanpercentile(boot_aucs, low_p))
    auc_high = float(np.nanpercentile(boot_aucs, high_p))

    rec_low = [float(np.nanpercentile(boot_recalls[c], low_p)) for c in range(n_classes)]
    rec_high = [float(np.nanpercentile(boot_recalls[c], high_p)) for c in range(n_classes)]

    return {
        "roc_auc_ovr_macro": {
            "point": point_auc,
            "low": auc_low,
            "high": auc_high,
            "n_boot": n,
        },
        "recall_per_class": {
            "point": point_rec,
            "low": rec_low,
            "high": rec_high,
            "by_class": [
                {"point": point_rec[c], "low": rec_low[c], "high": rec_high[c]}
                for c in range(n_classes)
            ],
            "n_boot": n,
        },
    }
