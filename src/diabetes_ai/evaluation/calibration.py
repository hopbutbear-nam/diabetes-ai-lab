"""Hiệu chuẩn xác suất đa lớp (Identity, Logit Multinomial, Isotonic OvR) và out-of-fold."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, Self

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from diabetes_ai.evaluation.metrics import brier_multiclass, ece_ovr


class BaseCalibrator(ABC):
    """Lớp cơ sở cho các bộ hiệu chuẩn xác suất."""

    method: str

    @abstractmethod
    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        """Huấn luyện bộ hiệu chuẩn dựa trên xác suất và nhãn thực tế."""
        pass

    @abstractmethod
    def predict(self, proba: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất đã hiệu chuẩn (tổng mỗi hàng bằng 1, kiểu float64)."""
        pass

    def predict_proba(self, proba: np.ndarray) -> np.ndarray:
        """Bí danh cho predict để tương thích với giao diện chuẩn."""
        return self.predict(proba)


class IdentityCalibrator(BaseCalibrator):
    """Bộ hiệu chuẩn danh nghĩa không thay đổi xác suất (none)."""

    method: str = "none"

    def __init__(self) -> None:
        self._fitted: bool = False

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        """Không cần huấn luyện tham số."""
        self._fitted = True
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        """Chuẩn hóa hàng tổng 1 và trả về mảng float64."""
        proba_arr = np.asarray(proba, dtype=np.float64)
        row_sums = proba_arr.sum(axis=1, keepdims=True)
        uniform = np.full_like(proba_arr, 1.0 / proba_arr.shape[1])
        normalized = np.where(row_sums == 0, uniform, proba_arr / row_sums)
        return normalized.astype(np.float64)


class LogitMultinomialCalibrator(BaseCalibrator):
    """Hiệu chuẩn hồi quy logistic đa lớp trên không gian logit."""

    method: str = "logit_multinomial"

    def __init__(self) -> None:
        self.lr: LogisticRegression | None = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        """Huấn luyện logistic regression đa lớp trên log-clip xác suất."""
        proba_arr = np.asarray(proba, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        self.lr = LogisticRegression(C=100.0, max_iter=2000)
        self.lr.fit(np.log(np.clip(proba_arr, 1e-6, 1.0)), y_arr)
        if list(self.lr.classes_) != list(range(proba_arr.shape[1])):
            raise ValueError("Thiếu lớp trong dữ liệu hiệu chuẩn")
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất đã hiệu chuẩn qua multinomial logistic regression."""
        if self.lr is None:
            raise RuntimeError("Calibrator chưa được huấn luyện (fit)")
        proba_arr = np.asarray(proba, dtype=np.float64)
        features = np.log(np.clip(proba_arr, 1e-6, 1.0))
        calibrated = self.lr.predict_proba(features).astype(np.float64)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        uniform = np.full_like(calibrated, 1.0 / calibrated.shape[1])
        normalized = np.where(row_sums == 0, uniform, calibrated / row_sums)
        return normalized.astype(np.float64)


class IsotonicOvRCalibrator(BaseCalibrator):
    """Hiệu chuẩn hồi quy đồng điệu từng lớp (One-vs-Rest) và chuẩn hóa hàng."""

    method: str = "isotonic_ovr"

    def __init__(self) -> None:
        self.models: list[IsotonicRegression] = []

    def fit(self, proba: np.ndarray, y: np.ndarray) -> Self:
        """Huấn luyện IsotonicRegression cho từng lớp."""
        proba_arr = np.asarray(proba, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        K = proba_arr.shape[1]
        self.models = []
        for k in range(K):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            y_bin = (y_arr == k).astype(np.float64)
            iso.fit(proba_arr[:, k], y_bin)
            self.models.append(iso)
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        """Dự đoán từng lớp, cộng 1e-9 và chuẩn hóa hàng tổng 1."""
        proba_arr = np.asarray(proba, dtype=np.float64)
        K = proba_arr.shape[1]
        if len(self.models) != K:
            raise ValueError(
                f"Số lượng mô hình ({len(self.models)}) không khớp với số lớp đầu vào ({K})"
            )
        cols = [self.models[k].predict(proba_arr[:, k]) for k in range(K)]
        out = np.column_stack(cols) + 1e-9
        row_sums = out.sum(axis=1, keepdims=True)
        uniform = np.full_like(out, 1.0 / K)
        normalized = np.where(row_sums == 0, uniform, out / row_sums)
        return normalized.astype(np.float64)


def fit_calibrator(method: str, proba: np.ndarray, y: np.ndarray) -> BaseCalibrator:
    """Khởi tạo và huấn luyện bộ hiệu chuẩn theo phương pháp tương ứng.

    Args:
        method: Phương pháp ('none', 'logit_multinomial', 'isotonic_ovr').
        proba: Xác suất dự đoán ban đầu (n, K).
        y: Nhãn thực tế (n,).

    Returns:
        BaseCalibrator: Đối tượng hiệu chuẩn đã fit.
    """
    calibrator: BaseCalibrator
    if method in ("none", "identity"):
        calibrator = IdentityCalibrator()
    elif method in ("logit_multinomial", "platt"):
        calibrator = LogitMultinomialCalibrator()
    elif method in ("isotonic_ovr", "isotonic"):
        calibrator = IsotonicOvRCalibrator()
    else:
        raise ValueError(
            f"Phương pháp hiệu chuẩn không hợp lệ: '{method}'. "
            "Chỉ hỗ trợ: 'none', 'logit_multinomial', 'isotonic_ovr'."
        )
    return calibrator.fit(proba, y)


def oof_calibrated(
    method: str,
    proba: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Tạo xác suất out-of-fold hiệu chuẩn qua StratifiedKFold.

    Mỗi dòng được hiệu chuẩn bởi bộ mô hình không thấy dòng dữ liệu đó.

    Args:
        method: Phương pháp hiệu chuẩn ('none', 'logit_multinomial', 'isotonic_ovr').
        proba: Ma trận xác suất ban đầu (n, K).
        y: Nhãn thực tế (n,).
        n_splits: Số fold cross-validation (mặc định 5).
        seed: Seed sinh chia fold (mặc định 42).

    Returns:
        np.ndarray: Ma trận xác suất OOF đã hiệu chuẩn (n, K).
    """
    proba_arr = np.asarray(proba, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)

    if method in ("none", "identity"):
        return IdentityCalibrator().fit(proba_arr, y_arr).predict(proba_arr)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros_like(proba_arr, dtype=np.float64)

    for train_idx, val_idx in skf.split(proba_arr, y_arr):
        cal = fit_calibrator(method, proba_arr[train_idx], y_arr[train_idx])
        oof_proba[val_idx] = cal.predict(proba_arr[val_idx])

    row_sums = oof_proba.sum(axis=1, keepdims=True)
    uniform = np.full_like(oof_proba, 1.0 / oof_proba.shape[1])
    normalized = np.where(row_sums == 0, uniform, oof_proba / row_sums)
    return normalized.astype(np.float64)


def select_calibration_method(
    proba: np.ndarray,
    y: np.ndarray,
    candidates: Sequence[str] = ("none", "logit_multinomial", "isotonic_ovr"),
    n_bins: int = 10,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Đánh giá và lựa chọn phương pháp hiệu chuẩn tốt nhất out-of-fold.

    Tiêu chí: chọn phương pháp có ECE macro OOF thấp nhất; nếu hòa chọn Brier OOF thấp hơn.

    Args:
        proba: Ma trận xác suất dự đoán (n, K).
        y: Nhãn thực tế (n,).
        candidates: Danh sách các phương pháp ứng viên.
        n_bins: Số bin cho ECE (mặc định 10).
        n_splits: Số fold OOF (mặc định 5).
        seed: Seed phân tầng KFold (mặc định 42).

    Returns:
        dict: Chứa 'chosen', 'ece_macro_before', 'brier_before' và chỉ số của từng ứng viên.
    """
    proba_arr = np.asarray(proba, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)

    ece_before = float(ece_ovr(y_arr, proba_arr, n_bins=n_bins)["macro"])
    brier_before = float(brier_multiclass(y_arr, proba_arr))

    cand_results: dict[str, dict[str, float]] = {}
    for cand in candidates:
        oof_p = oof_calibrated(cand, proba_arr, y_arr, n_splits=n_splits, seed=seed)
        ece_c = float(ece_ovr(y_arr, oof_p, n_bins=n_bins)["macro"])
        brier_c = float(brier_multiclass(y_arr, oof_p))
        cand_results[cand] = {
            "ece_macro_oof": ece_c,
            "brier_oof": brier_c,
        }

    # Chọn ECE macro thấp nhất, hòa thì Brier thấp hơn
    chosen = min(
        candidates,
        key=lambda c: (cand_results[c]["ece_macro_oof"], cand_results[c]["brier_oof"]),
    )

    result: dict[str, Any] = {
        "chosen": chosen,
        "ece_macro_before": ece_before,
        "brier_before": brier_before,
        "candidates": cand_results,
    }
    for cand, metrics in cand_results.items():
        result[cand] = metrics

    return result


def save_calibrator(cal: BaseCalibrator, path: str | Path) -> None:
    """Lưu calibrator xuống tệp bằng joblib.

    Args:
        cal: Đối tượng calibrator đã fit.
        path: Đường dẫn tệp cần lưu.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, save_path)


def load_calibrator(path: str | Path) -> BaseCalibrator:
    """Nạp calibrator từ tệp bằng joblib.

    Args:
        path: Đường dẫn tệp calibrator.

    Returns:
        BaseCalibrator: Calibrator đã nạp.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file calibrator tại: {load_path}")
    return joblib.load(load_path)
