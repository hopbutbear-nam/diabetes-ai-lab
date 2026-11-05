"""Mô hình XGBoost cho dữ liệu dạng bảng với cơ chế dự phòng HistGradientBoosting.

Tự động chuyển đổi giữa XGBoost và HistGradientBoostingClassifier tùy môi trường.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
from packaging.version import Version

from ..logging_utils import get_logger
from .base import BaseModel, ensure_proba

logger = get_logger(__name__)


def check_env_xgboost_ok() -> bool:
    """Kiểm tra cờ xgboost_ok trong env_check.json nếu tồn tại."""
    candidates = [
        Path("reports/metrics/env_check.json"),
        Path("env_check.json"),
    ]
    try:
        from ..paths import ROOT
        candidates.extend([
            ROOT / "reports" / "metrics" / "env_check.json",
            ROOT / "env_check.json",
        ])
    except Exception:
        pass

    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "xgboost_ok" in data and not data["xgboost_ok"]:
                    return False
            except Exception:
                pass
    return True


class XGBoostModel(BaseModel):
    """Mô hình XGBoost phân loại với backend 'xgboost' hoặc 'hgb'."""

    name: ClassVar[str] = "xgboost"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_classes: int = 3,
        seed: int = 42,
        backend: str = "auto",
    ) -> None:
        super().__init__(params=params, n_classes=n_classes, seed=seed)
        # Cho phép cấu hình backend qua params hoặc đối số trực tiếp
        raw_backend = self.params.pop("backend", backend)
        self.backend: str = raw_backend
        self.inner_model: Any = None

    def _resolve_backend(self, backend: str) -> str:
        """Xác định backend thực tế sẽ được sử dụng."""
        if backend == "auto":
            if not check_env_xgboost_ok():
                logger.warning(
                    "env_check.json có xgboost_ok=false. Dùng HistGradientBoostingClassifier (A12)."
                )
                return "hgb"
            try:
                import xgboost as xgb
                v = Version(xgb.__version__)
                if v >= Version("2.0.0"):
                    return "xgboost"
                else:
                    logger.warning(
                        "Phiên bản xgboost %s < 2.0.0. Dùng HistGradientBoostingClassifier làm backend dự phòng.",
                        xgb.__version__,
                    )
                    return "hgb"
            except Exception as e:
                logger.warning(
                    "Không thể nạp xgboost (%s). Dùng HistGradientBoostingClassifier làm backend dự phòng.",
                    e,
                )
                return "hgb"
        elif backend == "xgboost":
            return "xgboost"
        elif backend == "hgb":
            return "hgb"
        else:
            raise ValueError(
                f"Backend không hợp lệ: '{backend}'. Chỉ chấp nhận 'auto', 'xgboost', 'hgb'."
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> XGBoostModel:
        """Huấn luyện mô hình XGBoost hoặc HistGradientBoosting."""
        backend = self._resolve_backend(self.backend)
        self.fit_info["backend"] = backend

        if backend == "xgboost":
            import xgboost as xgb

            xgb_kwargs: dict[str, Any] = {
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": self.seed,
                "objective": "multi:softprob" if self.n_classes > 2 else "binary:logistic",
                "eval_metric": "auc",
            }
            # Cập nhật từ self.params
            xgb_kwargs.update(self.params)

            has_val = X_val is not None and y_val is not None
            if has_val:
                eval_set = [(X_val, y_val)]
                if "early_stopping_rounds" not in xgb_kwargs:
                    xgb_kwargs["early_stopping_rounds"] = 20
            else:
                eval_set = None
                xgb_kwargs.pop("early_stopping_rounds", None)

            model = xgb.XGBClassifier(**xgb_kwargs)
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                verbose=False,
            )
            self.inner_model = model

            if getattr(model, "best_iteration", None) is not None:
                self.fit_info["best_iteration"] = int(model.best_iteration)
            logger.info("XGBoostModel fit thành công với backend 'xgboost'.")

        else:
            # Nhánh HistGradientBoostingClassifier
            from sklearn.ensemble import HistGradientBoostingClassifier

            n_estimators = self.params.get("n_estimators", self.params.get("max_iter", 100))
            max_depth = self.params.get("max_depth", None)
            learning_rate = self.params.get("learning_rate", 0.1)

            ignored_keys = [
                "subsample",
                "colsample_bytree",
                "early_stopping_rounds",
                "tree_method",
                "eval_metric",
                "n_jobs",
                "objective",
            ]
            self.fit_info["ignored_params"] = [k for k in ignored_keys if k in self.params]

            hgb = HistGradientBoostingClassifier(
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                early_stopping=False,
                random_state=self.seed,
            )
            hgb.fit(X_train, y_train, sample_weight=sample_weight)
            self.inner_model = hgb

            if hasattr(hgb, "n_iter_"):
                self.fit_info["best_iteration"] = int(hgb.n_iter_)
            logger.info("XGBoostModel fit thành công với backend 'hgb'.")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán ma trận xác suất (n_samples, n_classes)."""
        if self.inner_model is None:
            raise RuntimeError("Mô hình XGBoostModel chưa được fit.")

        raw_proba = self.inner_model.predict_proba(X)
        if raw_proba.shape[1] == self.n_classes:
            return ensure_proba(raw_proba, self.n_classes)

        proba = np.zeros((len(X), self.n_classes), dtype=np.float64)
        classes = getattr(self.inner_model, "classes_", np.arange(raw_proba.shape[1]))
        for col_idx, cls_idx in enumerate(classes):
            if cls_idx < self.n_classes:
                proba[:, int(cls_idx)] = raw_proba[:, col_idx]

        return ensure_proba(proba, self.n_classes)

    def save(self, run_dir: Path) -> None:
        """Lưu mô hình vào run_dir / model.joblib."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "model.joblib"
        joblib.dump(self, path)
        logger.info("Đã lưu XGBoostModel tại %s", path)

    @classmethod
    def load(cls, run_dir: Path) -> XGBoostModel:
        """Tải mô hình từ run_dir / model.joblib."""
        run_dir = Path(run_dir)
        path = run_dir / "model.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Đối tượng nạp từ {path} không phải {cls.__name__}")
        logger.info("Đã nạp XGBoostModel từ %s", path)
        return model
