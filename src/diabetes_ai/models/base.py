"""Giao diện chung: fit, predict_proba, save, load.

Cung cấp lớp trừu tượng BaseModel và hàm ensure_proba.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


def ensure_proba(p: np.ndarray, n_classes: int) -> np.ndarray:
    """Validate và chuẩn hoá ma trận xác suất.

    - Kiểm tra shape là 2D (n_samples, n_classes), sai shape báo ValueError.
    - Đảm bảo mọi giá trị >= 0 (cắt âm).
    - Chuẩn hoá lại mỗi hàng sao cho tổng bằng 1.0 ± 1e-6.

    Args:
        p: Mảng xác suất đầu vào.
        n_classes: Số lượng lớp phân loại.

    Returns:
        np.ndarray: Ma trận xác suất float64 kích thước (n, n_classes).
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(
            f"Kích thước ma trận xác suất {p.shape} không khớp với n_classes={n_classes}."
        )

    p = np.clip(p, 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    zero_mask = (row_sums == 0).squeeze(axis=1)
    if np.any(zero_mask):
        logger.warning("Một hoặc nhiều hàng có tổng 0; sẽ phân phối đều 1/n_classes.")
        p[zero_mask] = 1.0 / n_classes
        row_sums = p.sum(axis=1, keepdims=True)

    p = p / row_sums
    return p


class BaseModel(ABC):
    """Lớp cơ sở trừu tượng cho mọi mô hình trong hệ thống."""

    name: ClassVar[str] = "base"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_classes: int = 3,
        seed: int = 42,
    ) -> None:
        """Khởi tạo mô hình cơ sở."""
        self.params: dict[str, Any] = params.copy() if params else {}
        self.n_classes: int = n_classes
        self.seed: int = seed
        self.fit_info: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> BaseModel:
        """Huấn luyện mô hình và trả về chính self."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Trả về ma trận xác suất (n_samples, n_classes) kiểu float64."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn dựa trên argmax của predict_proba."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1).astype(np.int64)

    @abstractmethod
    def save(self, run_dir: Path) -> None:
        """Lưu artifact mô hình vào thư mục run_dir."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, run_dir: Path) -> BaseModel:
        """Tải mô hình từ thư mục run_dir."""
        ...
