"""Mô-đun xuất gói mô hình suy luận tự đủ (inference bundle) và NumpyMLP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from diabetes_ai.config import load_config
from diabetes_ai.data.schema import FEATURES, class_names as get_class_names
from diabetes_ai.features.build_features import load_experiment_data
from diabetes_ai.models.registry import load_run
from diabetes_ai.paths import ROOT
from diabetes_ai.safety.disclaimers import DISCLAIMER_VI


class NumpyMLP:
    """Mô hình Perceptron đa tầng (MLP) thuần NumPy, không phụ thuộc vào PyTorch."""

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """Khởi tạo với danh sách trọng số và bias của các tầng Linear.

        Args:
            weights: Danh sách ma trận trọng số (out_features, in_features).
            biases: Danh sách vector bias (out_features,).
        """
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]

    @classmethod
    def from_torch(cls, model: Any) -> "NumpyMLP":
        """Trích xuất tham số từ mô hình MLPModel hoặc MLPNet của PyTorch.

        Args:
            model: Thực thể MLPModel hoặc MLPNet có phương thức export_numpy().

        Returns:
            NumpyMLP: Mô hình suy luận thuần NumPy.

        Raises:
            TypeError: Nếu mô hình không hỗ trợ xuất sang NumPy.
        """
        if hasattr(model, "export_numpy"):
            data = model.export_numpy()
        elif hasattr(model, "net") and hasattr(model.net, "export_numpy"):
            data = model.net.export_numpy()
        else:
            raise TypeError("Mô hình không hỗ trợ xuất sang NumPy (thiếu export_numpy).")
        return cls(weights=data["weights"], biases=data["biases"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán ma trận xác suất sử dụng kích hoạt ReLU và Softmax trừ max ổn định số học.

        Args:
            X: Mảng đặc trưng đầu vào kích thước (n_samples, n_features) hoặc (n_features,).

        Returns:
            np.ndarray: Ma trận xác suất kích thước (n_samples, n_classes).
        """
        x = np.asarray(X, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h = x
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w.T + b
            if i < n_layers - 1:
                h = np.maximum(0.0, z)
            else:
                h = z

        # Softmax ổn định số học bằng trừ max
        shift_h = h - np.max(h, axis=-1, keepdims=True)
        exp_h = np.exp(shift_h)
        probs = exp_h / np.sum(exp_h, axis=-1, keepdims=True)
        return probs.astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn theo argmax của xác suất.

        Args:
            X: Mảng đặc trưng đầu vào.

        Returns:
            np.ndarray: Mảng nhãn dự đoán 1D.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)


def make_bundle(
    model: Any,
    scaler: Any,
    class_names: list[str],
    task: str,
    variant: str,
    run_name: str,
    calibrator: Any = None,
    scale_factors: Any = None,
    threshold: float | None = None,
    pop_mean_raw: np.ndarray | list[float] | None = None,
    risk_levels: dict[str, float] | None = None,
    what_if_cfg: dict[str, Any] | None = None,
    metrics_val: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Tạo từ điển bundle tự đủ chứa toàn bộ artifact phục vụ suy luận.

    Args:
        model: Đối tượng mô hình (NumpyMLP, XGBModel, v.v.).
        scaler: Đối tượng StandardScalerNP hoặc dict tham số scaler.
        class_names: Danh sách tên lớp.
        task: Tác vụ ('multiclass' hoặc 'binary').
        variant: Biến thể ('full' hoặc 'dedup').
        run_name: Tên của lần chạy huấn luyện.
        calibrator: Bộ hiệu chuẩn xác suất (tùy chọn).
        scale_factors: Hệ số tỉ lệ ngưỡng đa lớp (tùy chọn).
        threshold: Ngưỡng quyết định nhị phân (tùy chọn).
        pop_mean_raw: Vector trung bình đặc trưng thô của quần thể (tùy chọn).
        risk_levels: Cấu hình phân ngưỡng mức nguy cơ (tùy chọn).
        what_if_cfg: Cấu hình kịch bản what-if (tùy chọn).
        metrics_val: Số liệu đo trên tập validation (tùy chọn).
        backend: Tên backend mô hình (tùy chọn).

    Returns:
        dict[str, Any]: Từ điển bundle hoàn chỉnh theo lược đồ schema_version=1.
    """
    scaler_dict: dict[str, Any]
    if hasattr(scaler, "to_dict"):
        scaler_dict = scaler.to_dict()
    elif isinstance(scaler, dict):
        scaler_dict = scaler
    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        scaler_dict = {
            "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "std": np.asarray(scaler.scale_, dtype=float).tolist(),
            "feature_names": getattr(scaler, "feature_names_in_", list(FEATURES)),
        }
    else:
        scaler_dict = scaler

    # Xác định pop_mean_raw nếu chưa được truyền
    resolved_pop_mean: np.ndarray | None = None
    if pop_mean_raw is not None:
        resolved_pop_mean = np.asarray(pop_mean_raw, dtype=np.float32)
    elif hasattr(scaler, "mean_") and scaler.mean_ is not None:
        resolved_pop_mean = np.asarray(scaler.mean_, dtype=np.float32)
    elif isinstance(scaler, dict) and "mean" in scaler:
        resolved_pop_mean = np.asarray(scaler["mean"], dtype=np.float32)

    # Nạp cấu hình mặc định nếu None
    if risk_levels is None:
        try:
            risk_levels = load_config("serve").get("risk_levels", {"low": 0.10, "medium": 0.25})
        except Exception:
            risk_levels = {"low": 0.10, "medium": 0.25}

    if what_if_cfg is None:
        try:
            what_if_cfg = load_config("features").get("what_if", {})
        except Exception:
            what_if_cfg = {}

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "task": task,
        "variant": variant,
        "class_names": list(class_names),
        "feature_order": list(FEATURES),
        "scaler": scaler_dict,
        "model": model,
        "calibrator": calibrator,
        "scale_factors": scale_factors,
        "threshold": threshold,
        "pop_mean_raw": resolved_pop_mean,
        "risk_levels": risk_levels,
        "what_if": what_if_cfg,
        "metrics_val": metrics_val,
        "trained_on": "BRFSS 2015 (CDC), 253.680 hồ sơ",
        "disclaimer": DISCLAIMER_VI,
    }
    if backend is not None:
        bundle["backend"] = backend

    return bundle


def save_bundle(bundle: dict[str, Any], path: Path | str) -> Path:
    """Lưu bundle suy luận ra tệp joblib.

    Args:
        bundle: Từ điển bundle chứa mô hình và siêu dữ liệu.
        path: Đường dẫn tệp đích.

    Returns:
        Path: Đường dẫn tệp đã lưu.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return out_path


def load_bundle(path: Path | str) -> dict[str, Any]:
    """Nạp bundle suy luận từ tệp joblib.

    Args:
        path: Đường dẫn tệp bundle.

    Returns:
        dict[str, Any]: Từ điển bundle đã nạp.

    Raises:
        FileNotFoundError: Nếu tệp không tồn tại.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file bundle tại {p}")
    return joblib.load(p)


def export_best(
    task: str = "multiclass",
    variant: str = "full",
    out: Path | str | None = None,
) -> Path:
    """Xuất mô hình tốt nhất theo model_selection.json thành gói bundle tự đủ.

    Args:
        task: Tác vụ phân loại ('multiclass' hoặc 'binary').
        variant: Biến thể dữ liệu ('full' hoặc 'dedup').
        out: Đường dẫn xuất (mặc định theo configs/serve.yaml hoặc models/best_model.joblib).

    Returns:
        Path: Đường dẫn tệp bundle đã lưu.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file model_selection.json.
        KeyError: Nếu task_variant không có trong model_selection.json.
    """
    if out is not None:
        out_path = Path(out)
    else:
        try:
            serve_cfg = load_config("serve")
            model_path = serve_cfg.get("model_path", "models/best_model.joblib")
            out_path = ROOT / model_path
        except Exception:
            out_path = ROOT / "models" / "best_model.joblib"

    selection_path = ROOT / "reports" / "metrics" / "model_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file {selection_path}. Hãy chạy scripts/calibrate_and_threshold.py trước."
        )

    with open(selection_path, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    key = f"{task}_{variant}"
    if key not in selection_data:
        raise KeyError(
            f"Không tìm thấy cấu hình '{key}' trong {selection_path}. Các khóa hiện có: {list(selection_data.keys())}"
        )

    best_run = selection_data[key]["best"]

    # Nạp run đã chọn
    run = load_run(best_run)

    # Chuyển đổi mô hình sang NumpyMLP nếu là MLP
    model = run.model
    backend = run.meta.get("backend", "sklearn")
    if (
        hasattr(model, "export_numpy")
        or getattr(model, "name", None) == "mlp"
        or run.meta.get("model_key") == "mlp"
    ):
        model = NumpyMLP.from_torch(model)
        backend = "numpy"

    scaler = run.scaler
    class_names = run.meta.get("class_names") or get_class_names(task)
    calibrator = run.calibrator
    scale_factors = run.scale_factors

    threshold = None
    if run.postproc and "thresholds" in run.postproc:
        threshold = run.postproc["thresholds"].get("threshold")

    # Xác định pop_mean_raw
    pop_mean_raw = None
    if hasattr(scaler, "mean_") and scaler.mean_ is not None:
        pop_mean_raw = np.asarray(scaler.mean_, dtype=np.float32)
    elif isinstance(scaler, dict) and "mean" in scaler:
        pop_mean_raw = np.asarray(scaler["mean"], dtype=np.float32)
    else:
        exp_data = load_experiment_data(task=task, variant=variant, include_test=False)
        pop_mean_raw = exp_data.pop_mean_raw

    # Nạp thông tin đánh giá validation nếu có
    metrics_val = None
    train_metric_path = ROOT / "reports" / "metrics" / f"train_{best_run}.json"
    if train_metric_path.is_file():
        try:
            with open(train_metric_path, "r", encoding="utf-8") as f:
                tdata = json.load(f)
                metrics_val = tdata.get("val")
        except Exception:
            metrics_val = None

    bundle = make_bundle(
        model=model,
        scaler=scaler,
        class_names=class_names,
        task=task,
        variant=variant,
        run_name=best_run,
        calibrator=calibrator,
        scale_factors=scale_factors,
        threshold=threshold,
        pop_mean_raw=pop_mean_raw,
        metrics_val=metrics_val,
        backend=backend,
    )

    save_bundle(bundle, out_path)
    return out_path
