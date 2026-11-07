"""Tính chỉ số nguy cơ và xác định mức phân tầng nguy cơ kết hợp cảnh báo an toàn."""

from __future__ import annotations

from typing import Any

import numpy as np

from diabetes_ai.safety.rules import Alert


def risk_index(proba: np.ndarray | list[float], task: str = "multiclass") -> float:
    """Tính chỉ số nguy cơ tổng hợp r từ phân phối xác suất dự đoán của mô hình.

    Quy ước:
    - Bài toán đa lớp: r = P(Tiểu đường) + 0.5 * P(Tiền tiểu đường) = P2 + 0.5 * P1.
    - Bài toán nhị phân: r = P(Dương tính) = P1.

    Args:
        proba: Mảng 1D hoặc 2D chứa xác suất các lớp.
        task: Loại bài toán ('multiclass' hoặc 'binary').

    Returns:
        float: Chỉ số nguy cơ trong khoảng [0.0, 1.0].
    """
    arr = np.asarray(proba, dtype=float).ravel()
    task_normalized = str(task).lower()

    if task_normalized == "binary":
        val = float(arr[1]) if len(arr) > 1 else float(arr[0])
    elif task_normalized == "multiclass":
        if len(arr) >= 3:
            val = float(arr[2] + 0.5 * arr[1])
        elif len(arr) == 2:
            val = float(arr[1])
        else:
            val = float(arr[0])
    else:
        if len(arr) >= 3:
            val = float(arr[2] + 0.5 * arr[1])
        elif len(arr) == 2:
            val = float(arr[1])
        else:
            val = float(arr[0])

    return float(np.clip(val, 0.0, 1.0))


def level_from_index(r: float, low: float = 0.10, medium: float = 0.25) -> str:
    """Xác định mức nguy cơ từ chỉ số r dựa trên các ngưỡng phân tầng.

    Args:
        r: Chỉ số nguy cơ (0.0 đến 1.0).
        low: Ngưỡng mức thấp (mặc định 0.10).
        medium: Ngưỡng mức trung bình (mặc định 0.25).

    Returns:
        str: 'thấp' nếu r < low, 'trung bình' nếu low <= r < medium, 'cao' nếu r >= medium.
    """
    if r < low:
        return "thấp"
    if r < medium:
        return "trung bình"
    return "cao"


def combine(
    r: float,
    alerts: list[Alert | Any],
    low: float = 0.10,
    medium: float = 0.25,
) -> str:
    """Ghép chỉ số nguy cơ của mô hình với các cảnh báo an toàn.

    Quy tắc an toàn:
    - Nếu có cảnh báo mức 'danger': mức cuối cùng luôn là 'cao'.
    - Nếu có cảnh báo mức 'warn' và mức ban đầu là 'thấp': nâng lên 'trung bình'.
    - Còn lại: giữ nguyên mức tính từ chỉ số r.

    Args:
        r: Chỉ số nguy cơ.
        alerts: Danh sách các cảnh báo luật an toàn kích hoạt.
        low: Ngưỡng mức thấp.
        medium: Ngưỡng mức trung bình.

    Returns:
        str: Mức nguy cơ tổng hợp ('thấp', 'trung bình', hoặc 'cao').
    """
    base = level_from_index(r, low=low, medium=medium)
    has_danger = any(getattr(a, "level", None) == "danger" for a in alerts)
    has_warn = any(getattr(a, "level", None) == "warn" for a in alerts)

    if has_danger:
        return "cao"
    if has_warn and base == "thấp":
        return "trung bình"
    return base
