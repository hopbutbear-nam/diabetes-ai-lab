"""Registry đăng ký và tải mô hình huấn luyện.

Cung cấp:
- MODEL_REGISTRY: bảng ánh xạ tên mô hình -> lớp mô hình.
- build_model: khởi tạo mô hình theo tên và tham số.
- LoadedRun: đối tượng chứa mô hình, scaler, meta và các hàm suy luận.
- load_run: nạp toàn bộ artifact của một lần chạy (run).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..features.scalers import StandardScalerNP
from ..logging_utils import get_logger
from ..paths import METRICS_DIR, MODELS_DIR
from .base import BaseModel
from .baselines import LogisticRegressionModel, MajorityClassifier
from .xgboost_model import XGBoostModel

logger = get_logger(__name__)

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "majority": MajorityClassifier,
    "logreg": LogisticRegressionModel,
    "xgboost": XGBoostModel,
}


def build_model(
    model_key: str,
    params: dict[str, Any] | None = None,
    n_classes: int = 3,
    seed: int = 42,
) -> BaseModel:
    """Khởi tạo đối tượng mô hình theo tên định danh trong MODEL_REGISTRY.

    Args:
        model_key: Khóa nhận diện mô hình ('majority', 'logreg', 'xgboost', ...).
        params: Siêu tham số truyền vào mô hình.
        n_classes: Số lượng lớp bài toán (mặc định 3).
        seed: Seed sinh số ngẫu nhiên.

    Returns:
        BaseModel: Thực thể mô hình kế thừa từ BaseModel.

    Raises:
        ValueError: Nếu model_key không nằm trong danh sách hợp lệ.
    """
    model_params = params.copy() if params else {}

    # Hỗ trợ MLP với import trễ để không bắt buộc PyTorch khi chỉ chạy baseline / XGBoost
    if model_key == "mlp":
        if "mlp" in MODEL_REGISTRY:
            model_cls = MODEL_REGISTRY["mlp"]
        else:
            try:
                from .mlp import MLPModel

                model_cls = MLPModel
            except ImportError as err:
                raise ValueError(
                    f"Không thể nạp mô hình 'mlp': chưa cài đặt hoặc thiếu phụ thuộc ({err})."
                ) from err
        return model_cls(params=model_params, n_classes=n_classes, seed=seed)

    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Mô hình '{model_key}' không có. Hợp lệ: {sorted(MODEL_REGISTRY)}"
        )

    model_cls = MODEL_REGISTRY[model_key]
    return model_cls(params=model_params, n_classes=n_classes, seed=seed)


@dataclass
class LoadedRun:
    """Đối tượng đóng gói một lần chạy đã huấn luyện kèm đầy đủ công cụ suy luận."""

    run_name: str
    run_dir: Path
    model: BaseModel
    scaler: StandardScalerNP
    meta: dict[str, Any]
    calibrator: Any | None = None
    postproc: dict[str, Any] | None = None

    def predict_proba_scaled(
        self, Xs: np.ndarray, calibrated: bool = True
    ) -> np.ndarray:
        """Dự đoán xác suất trên ma trận đặc trưng đã chuẩn hoá (scaled).

        Args:
            Xs: Ma trận đặc trưng đã qua scaler.
            calibrated: Nếu True và có calibrator thì áp dụng hiệu chuẩn.

        Returns:
            np.ndarray: Ma trận xác suất kích thước (n_samples, n_classes).
        """
        proba = self.model.predict_proba(Xs)
        if calibrated and self.calibrator is not None:
            if hasattr(self.calibrator, "predict"):
                proba = self.calibrator.predict(proba)
            elif hasattr(self.calibrator, "predict_proba"):
                proba = self.calibrator.predict_proba(proba)
            elif callable(self.calibrator):
                proba = self.calibrator(proba)
        return proba

    def predict_proba(
        self, X_raw: np.ndarray, calibrated: bool = True
    ) -> np.ndarray:
        """Dự đoán xác suất từ đặc trưng thô (chưa scale).

        Args:
            X_raw: Ma trận đặc trưng thô.
            calibrated: Nếu True và có calibrator thì áp dụng hiệu chuẩn.

        Returns:
            np.ndarray: Ma trận xác suất kích thước (n_samples, n_classes).
        """
        Xs = self.scaler.transform(X_raw)
        return self.predict_proba_scaled(Xs, calibrated=calibrated)

    @property
    def scale_factors(self) -> np.ndarray | None:
        """Lấy hệ số tỉ lệ ngưỡng từ thông tin postproc nếu có."""
        if self.postproc:
            th = self.postproc.get("thresholds", {})
            if "scale" in th and th["scale"] is not None:
                return np.asarray(th["scale"], dtype=np.float64)
            if "scale_factors" in th and th["scale_factors"] is not None:
                return np.asarray(th["scale_factors"], dtype=np.float64)
            if "scale_factors" in self.postproc and self.postproc["scale_factors"] is not None:
                return np.asarray(self.postproc["scale_factors"], dtype=np.float64)
        return None

    def predict_labels(self, proba: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn từ ma trận xác suất (áp dụng hệ số tỉ lệ/ngưỡng nếu có, không thì argmax).

        Args:
            proba: Ma trận xác suất.

        Returns:
            np.ndarray: Mảng nhãn số nguyên 1D (n_samples,).
        """
        proba = np.asarray(proba, dtype=np.float64)

        if self.postproc and "thresholds" in self.postproc:
            th = self.postproc["thresholds"]
            if "scale" in th and th["scale"] is not None:
                scale = np.asarray(th["scale"], dtype=np.float64)
                if scale.shape[0] == proba.shape[-1]:
                    scaled = proba * scale
                    return np.argmax(scaled, axis=-1).astype(np.int64)
            if "threshold" in th and th["threshold"] is not None:
                threshold = float(th["threshold"])
                if proba.ndim == 2:
                    col = 1 if proba.shape[1] > 1 else 0
                    return (proba[:, col] >= threshold).astype(np.int64)
                return (proba >= threshold).astype(np.int64)

        if self.scale_factors is not None:
            scaled = proba * self.scale_factors
            return np.argmax(scaled, axis=-1).astype(np.int64)

        if proba.ndim == 2:
            return np.argmax(proba, axis=-1).astype(np.int64)
        return (proba >= 0.5).astype(np.int64)


def load_run(run_name: str, models_dir: Path | None = None) -> LoadedRun:
    """Nạp một lần chạy đã huấn luyện từ thư mục models/<run_name>.

    Đọc meta.json, scaler.json, file mô hình; nếu có calibrator.joblib thì nạp;
    nếu có postproc_<run_name>.json thì nạp.

    Args:
        run_name: Tên của lần chạy (ví dụ 'logreg_multiclass_full').
        models_dir: Thư mục gốc chứa các run (mặc định 'models/').

    Returns:
        LoadedRun: Thực thể chứa toàn bộ artifact đã nạp.

    Raises:
        FileNotFoundError: Nếu thư mục run hoặc file bắt buộc không tồn tại.
        ValueError: Nếu meta.json không hợp lệ.
    """
    base_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    run_dir = base_dir / run_name

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Chưa có run '{run_name}'. Hãy chạy scripts/train.py")

    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file meta.json tại {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scaler_path = run_dir / "scaler.json"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file scaler.json tại {scaler_path}")
    scaler = StandardScalerNP.load_json(scaler_path)

    model_key = meta.get("model_key") or meta.get("model")
    if not model_key:
        raise ValueError(f"meta.json tại {meta_path} không chứa trường 'model_key'.")

    if model_key == "mlp":
        if "mlp" in MODEL_REGISTRY:
            model_cls = MODEL_REGISTRY["mlp"]
        else:
            from .mlp import MLPModel

            model_cls = MLPModel
    elif model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Mô hình '{model_key}' không có. Hợp lệ: {sorted(MODEL_REGISTRY)}"
        )
    else:
        model_cls = MODEL_REGISTRY[model_key]

    model = model_cls.load(run_dir)

    calibrator = None
    calibrator_path = run_dir / "calibrator.joblib"
    if calibrator_path.is_file():
        import joblib

        calibrator = joblib.load(calibrator_path)
        logger.info("Đã nạp calibrator từ %s", calibrator_path)

    postproc = None
    candidate_postprocs = [
        METRICS_DIR / f"postproc_{run_name}.json",
        run_dir.parent.parent / "reports" / "metrics" / f"postproc_{run_name}.json",
        Path("reports/metrics") / f"postproc_{run_name}.json",
        run_dir / f"postproc_{run_name}.json",
    ]
    for p in candidate_postprocs:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    postproc = json.load(f)
                logger.info("Đã nạp postproc từ %s", p)
                break
            except Exception as e:
                logger.warning("Không thể đọc file postproc %s: %s", p, e)

    return LoadedRun(
        run_name=run_name,
        run_dir=run_dir,
        model=model,
        scaler=scaler,
        meta=meta,
        calibrator=calibrator,
        postproc=postproc,
    )
