"""Majority class và Logistic Regression (mô hình đối chứng).

Cung cấp:
- MajorityClassifier: dự đoán phân phối tiên nghiệm của nhãn.
- LogisticRegressionModel: bọc sklearn.linear_model.LogisticRegression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from ..logging_utils import get_logger
from .base import BaseModel, ensure_proba

logger = get_logger(__name__)


class MajorityClassifier(BaseModel):
    """Mô hình dự đoán tần suất lớp trên tập huấn luyện làm xác suất."""

    name = "majority"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_classes: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__(params=params, n_classes=n_classes, seed=seed)
        self.prior_: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> MajorityClassifier:
        """Lưu tần suất các lớp trong tập huấn luyện làm prior_."""
        counts = np.bincount(y_train, weights=sample_weight, minlength=self.n_classes).astype(np.float64)
        total = counts.sum()
        if total > 0:
            self.prior_ = counts / total
        else:
            self.prior_ = np.ones(self.n_classes, dtype=np.float64) / self.n_classes

        self.fit_info["prior"] = self.prior_.tolist()
        logger.info("MajorityClassifier đã fit với prior: %s", self.prior_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Trả về mảng xác suất lặp lại prior_ cho mỗi mẫu."""
        if self.prior_ is None:
            raise RuntimeError("Mô hình MajorityClassifier chưa được fit.")
        n_samples = len(X)
        proba = np.tile(self.prior_, (n_samples, 1))
        return ensure_proba(proba, self.n_classes)

    def save(self, run_dir: Path) -> None:
        """Lưu mô hình MajorityClassifier thành file model.json."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "n_classes": self.n_classes,
            "seed": self.seed,
            "params": self.params,
            "prior_": self.prior_.tolist() if self.prior_ is not None else None,
            "fit_info": self.fit_info,
        }
        with open(run_dir / "model.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Đã lưu MajorityClassifier tại %s", run_dir / "model.json")

    @classmethod
    def load(cls, run_dir: Path) -> MajorityClassifier:
        """Tải mô hình MajorityClassifier từ model.json."""
        run_dir = Path(run_dir)
        path = run_dir / "model.json"
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        model = cls(
            params=payload.get("params"),
            n_classes=payload.get("n_classes", 3),
            seed=payload.get("seed", 42),
        )
        if payload.get("prior_") is not None:
            model.prior_ = np.array(payload["prior_"], dtype=np.float64)
        model.fit_info = payload.get("fit_info", {})
        logger.info("Đã nạp MajorityClassifier từ %s", path)
        return model


class LogisticRegressionModel(BaseModel):
    """Mô hình Logistic Regression bọc từ sklearn."""

    name = "logreg"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_classes: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__(params=params, n_classes=n_classes, seed=seed)
        lr_params = self.params.copy()
        # Đừng truyền multi_class; để mặc định đa lớp softmax
        lr_params.pop("multi_class", None)
        lr_params.setdefault("random_state", seed)
        self.sklearn_params = lr_params
        self.model: LogisticRegression = LogisticRegression(**self.sklearn_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> LogisticRegressionModel:
        """Huấn luyện Logistic Regression.

        Bỏ qua sample_weight nếu params đã có class_weight để tránh nhân đôi trọng số.
        Lưu n_iter_ vào fit_info và cảnh báo nếu n_iter_ >= max_iter.
        """
        has_class_weight = "class_weight" in self.sklearn_params and self.sklearn_params["class_weight"] is not None
        sw = None if has_class_weight else sample_weight

        self.model.fit(X_train, y_train, sample_weight=sw)

        if hasattr(self.model, "n_iter_"):
            n_iter = int(np.max(self.model.n_iter_))
            self.fit_info["n_iter_"] = n_iter
            self.fit_info["n_iter"] = n_iter
            max_iter = getattr(self.model, "max_iter", 100)
            if n_iter >= max_iter:
                logger.warning(
                    "LogisticRegression đạt số vòng lặp tối đa (n_iter_=%s >= max_iter=%s), mô hình có thể chưa hội tụ.",
                    n_iter,
                    max_iter,
                )

        logger.info("LogisticRegressionModel đã fit thành công.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán ma trận xác suất (n_samples, n_classes)."""
        if self.model is None:
            raise RuntimeError("Mô hình LogisticRegressionModel chưa được fit.")

        raw_proba = self.model.predict_proba(X)
        if raw_proba.shape[1] == self.n_classes:
            return ensure_proba(raw_proba, self.n_classes)

        proba = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for col_idx, cls_idx in enumerate(self.model.classes_):
            if cls_idx < self.n_classes:
                proba[:, int(cls_idx)] = raw_proba[:, col_idx]

        return ensure_proba(proba, self.n_classes)

    def save(self, run_dir: Path) -> None:
        """Lưu mô hình thành model.joblib."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "model.joblib"
        joblib.dump(self, path)
        logger.info("Đã lưu LogisticRegressionModel tại %s", path)

    @classmethod
    def load(cls, run_dir: Path) -> LogisticRegressionModel:
        """Tải mô hình từ model.joblib."""
        run_dir = Path(run_dir)
        path = run_dir / "model.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Đối tượng nạp từ {path} không phải {cls.__name__}")
        logger.info("Đã nạp LogisticRegressionModel từ %s", path)
        return model
