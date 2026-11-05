"""
Chọn hệ số ngưỡng (scale factors) và ngưỡng nhị phân (binary threshold)
chỉ trên tập VALIDATION — không được dùng tập test.

API công khai:
  apply_scale(proba, scale) -> np.ndarray
  select_scale_factors(y_val, proba_val, objective, grid) -> dict
  select_binary_threshold(y_val, p_pos, objective, precision_floor) -> dict
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------

def apply_scale(proba: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    """Áp hệ số nhân vào xác suất rồi lấy argmax theo cột.

    Parameters
    ----------
    proba : np.ndarray, shape (n, K)
        Ma trận xác suất (mỗi hàng tổng ≈ 1).
    scale : Sequence[float], độ dài K
        Hệ số nhân cho từng lớp.

    Returns
    -------
    np.ndarray, shape (n,) — nhãn dự đoán sau khi nhân hệ số.
    """
    scale_arr = np.asarray(scale, dtype=float)
    return np.argmax(proba * scale_arr, axis=1)


# ---------------------------------------------------------------------------
# Chọn hệ số ngưỡng (multiclass)
# ---------------------------------------------------------------------------

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def select_scale_factors(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    objective: str = "macro_f1",
    grid: dict | None = None,
) -> dict:
    """Tìm hệ số ngưỡng (s_pre, s_dia) tối đa hóa macro-F1 trên validation.

    Lớp 0 (không bệnh) cố định hệ số = 1.0.
    Lưới mặc định:
      s_pre ∈ geomspace(0.5, 30, 41)  — lớp 1 (tiền tiểu đường)
      s_dia ∈ geomspace(0.5, 8,  33)  — lớp 2 (tiểu đường)
    Điểm (1,1) luôn được thêm vào lưới để đảm bảo kết quả ≥ argmax thuần.

    Parameters
    ----------
    y_val      : nhãn thật (int) trên tập validation.
    proba_val  : ma trận xác suất (n, K) trên tập validation.
    objective  : chỉ hỗ trợ "macro_f1".
    grid       : dict tuỳ chọn với khoá "s_pre" và "s_dia" (array-like).

    Returns
    -------
    dict với các khoá:
      scale               — list[float] độ dài K (hệ số tối ưu)
      val_macro_f1_scaled — macro-F1 với hệ số tối ưu
      val_macro_f1_argmax — macro-F1 với argmax thuần (baseline)
      val_recall          — recall từng lớp (list)
      val_precision       — precision từng lớp (list)
    """
    if objective != "macro_f1":
        raise ValueError(f"objective '{objective}' chưa được hỗ trợ; chỉ 'macro_f1'.")

    y_val = np.asarray(y_val)
    proba_val = np.asarray(proba_val, dtype=float)
    n_classes = proba_val.shape[1]

    # Baseline: argmax thuần
    y_argmax = np.argmax(proba_val, axis=1)
    baseline_f1 = _macro_f1(y_val, y_argmax)

    # Lưới tìm kiếm
    if grid is not None:
        s_pre_grid = np.asarray(grid.get("s_pre", np.geomspace(0.5, 30, 41)))
        s_dia_grid = np.asarray(grid.get("s_dia", np.geomspace(0.5, 8, 33)))
    else:
        s_pre_grid = np.geomspace(0.5, 30, 41)
        s_dia_grid = np.geomspace(0.5, 8, 33)

    # Đảm bảo điểm (1, 1) luôn có trong lưới
    if not np.any(np.isclose(s_pre_grid, 1.0)):
        s_pre_grid = np.sort(np.append(s_pre_grid, 1.0))
    if not np.any(np.isclose(s_dia_grid, 1.0)):
        s_dia_grid = np.sort(np.append(s_dia_grid, 1.0))

    best_f1 = -1.0
    best_scale: list[float] = [1.0] * n_classes

    for s_pre in s_pre_grid:
        for s_dia in s_dia_grid:
            scale = [1.0, float(s_pre), float(s_dia)]
            if n_classes == 2:
                scale = [1.0, float(s_pre)]
            y_pred = apply_scale(proba_val, scale[:n_classes])
            f1 = _macro_f1(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_scale = scale[:n_classes]

    # Thống kê với hệ số tốt nhất
    y_best = apply_scale(proba_val, best_scale)
    labels = list(range(n_classes))
    recall = recall_score(y_val, y_best, labels=labels, average=None, zero_division=0).tolist()
    precision = precision_score(y_val, y_best, labels=labels, average=None, zero_division=0).tolist()

    return {
        "scale": best_scale,
        "val_macro_f1_scaled": best_f1,
        "val_macro_f1_argmax": baseline_f1,
        "val_recall": recall,
        "val_precision": precision,
    }


# ---------------------------------------------------------------------------
# Chọn ngưỡng nhị phân
# ---------------------------------------------------------------------------

def select_binary_threshold(
    y_val: np.ndarray,
    p_pos: np.ndarray,
    objective: str = "macro_f1",
    precision_floor: float = 0.30,
) -> dict:
    """Chọn ngưỡng nhị phân tối ưu trên tập validation.

    Hỗ trợ hai chế độ:
    - "macro_f1": tìm ngưỡng tối đa hóa macro-F1 trên lưới 0.01–0.99 (bước 0.01).
    - "recall_at_precision": tìm ngưỡng cho recall cao nhất với precision ≥
      precision_floor. Nếu không có ngưỡng nào đáp ứng → ValueError.

    Parameters
    ----------
    y_val           : nhãn thật nhị phân {0, 1}.
    p_pos           : xác suất lớp dương (shape n,).
    objective       : "macro_f1" hoặc "recall_at_precision".
    precision_floor : precision tối thiểu (chỉ dùng khi objective = "recall_at_precision").

    Returns
    -------
    dict với khoá:
      threshold       — float
      val_macro_f1    — float (macro-F1 tại ngưỡng chọn)
      val_recall      — float (recall lớp 1 tại ngưỡng chọn)
      val_precision   — float (precision lớp 1 tại ngưỡng chọn)
      objective       — str
    """
    y_val = np.asarray(y_val)
    p_pos = np.asarray(p_pos, dtype=float)

    thresholds = np.arange(0.01, 1.00, 0.01)

    if objective == "macro_f1":
        best_thr = 0.5
        best_f1 = -1.0
        for thr in thresholds:
            y_pred = (p_pos >= thr).astype(int)
            f1 = _macro_f1(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        y_pred_best = (p_pos >= best_thr).astype(int)
        rec = float(recall_score(y_val, y_pred_best, zero_division=0))
        prec = float(precision_score(y_val, y_pred_best, zero_division=0))
        return {
            "threshold": best_thr,
            "val_macro_f1": best_f1,
            "val_recall": rec,
            "val_precision": prec,
            "objective": objective,
        }

    elif objective == "recall_at_precision":
        best_thr = None
        best_rec = -1.0
        max_prec_seen = 0.0
        for thr in thresholds:
            y_pred = (p_pos >= thr).astype(int)
            prec = float(precision_score(y_val, y_pred, zero_division=0))
            rec = float(recall_score(y_val, y_pred, zero_division=0))
            max_prec_seen = max(max_prec_seen, prec)
            if prec >= precision_floor and rec > best_rec:
                best_rec = rec
                best_thr = float(thr)

        if best_thr is None:
            raise ValueError(
                f"Không có ngưỡng nào đạt precision ≥ {precision_floor}. "
                f"Precision cao nhất đạt được: {max_prec_seen:.4f}."
            )

        y_pred_best = (p_pos >= best_thr).astype(int)
        f1 = _macro_f1(y_val, y_pred_best)
        prec_best = float(precision_score(y_val, y_pred_best, zero_division=0))
        return {
            "threshold": best_thr,
            "val_macro_f1": f1,
            "val_recall": best_rec,
            "val_precision": prec_best,
            "objective": objective,
        }

    else:
        raise ValueError(
            f"objective '{objective}' không hợp lệ. "
            "Chọn 'macro_f1' hoặc 'recall_at_precision'."
        )
