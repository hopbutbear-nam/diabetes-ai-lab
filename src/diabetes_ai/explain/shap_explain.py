"""Giải thích mô hình cục bộ cho từng cá nhân dùng SHAP (tùy chọn)."""

from typing import Any, Callable
import numpy as np


def shap_available() -> bool:
    """Kiểm tra thư viện shap có sẵn trong môi trường hay không."""
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def explain_instance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    x: np.ndarray,
    features: list[str],
    class_index: int = -1,
    max_background: int = 100,
) -> dict[str, float] | None:
    """Tính giá trị SHAP cho một mẫu cá nhân x.

    Args:
        predict_fn: Hàm nhận ma trận đặc trưng và trả về ma trận xác suất.
        X_background: Tập mẫu nền dùng làm baseline phân phối (thường trích từ validation set).
        x: Vector đặc trưng cá nhân (1D hoặc 2D kích thước 1 x n_features).
        features: Danh sách tên đặc trưng theo thứ tự các cột.
        class_index: Chỉ số của lớp cần giải thích (mặc định -1: lớp nguy cơ cao nhất/cuối cùng).
        max_background: Số lượng mẫu nền tối đa dùng cho masker Independent để giới hạn thời gian tính.

    Returns:
        dict[str, float] | None: Từ điển ánh xạ tên đặc trưng sang giá trị SHAP,
            hoặc None nếu shap không được cài đặt hoặc có lỗi xảy ra.
    """
    if not shap_available():
        return None

    try:
        import shap

        bg = np.asarray(X_background, dtype=float)
        if len(bg) > max_background:
            bg = bg[:max_background]

        masker = shap.maskers.Independent(bg)
        explainer = shap.Explainer(predict_fn, masker)

        x_2d = np.atleast_2d(x).astype(float)
        sv = explainer(x_2d)
        values = sv.values

        if values.ndim == 3:
            # Kích thước: (n_samples, n_features, n_classes)
            vals = values[0, :, class_index]
        elif values.ndim == 2:
            # Kích thước: (n_samples, n_features)
            vals = values[0, :]
        else:
            vals = values

        return {feat: float(vals[i]) for i, feat in enumerate(features)}
    except Exception:
        return None
