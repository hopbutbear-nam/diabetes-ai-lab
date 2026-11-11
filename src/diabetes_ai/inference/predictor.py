"""Mô-đun Predictor nạp mô hình, dự đoán nguy cơ, áp dụng luật an toàn và giải thích what-if."""

from __future__ import annotations

from functools import lru_cache

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from diabetes_ai.config import get_env, load_config
from diabetes_ai.data.schema import FEATURES
from diabetes_ai.explain.what_if import (
    WHAT_IF_NOTE,
    feature_effects,
    what_if_custom,
    what_if_scenarios,
)
from diabetes_ai.features.scalers import StandardScalerNP
from diabetes_ai.inference.export import load_bundle
from diabetes_ai.inference.schemas import PatientProfile
from diabetes_ai.paths import ROOT
from diabetes_ai.safety.disclaimers import DISCLAIMER_VI, advice_for
from diabetes_ai.safety.risk_level import combine, risk_index
from diabetes_ai.safety.rules import Alert, Rule, evaluate, load_rules


class ModelNotReady(RuntimeError):
    """Ngoại lệ ném ra khi tệp mô hình chưa sẵn sàng hoặc chưa được xuất."""


def _warn_if_env_mismatch(bundle: dict[str, Any]) -> None:
    """Cảnh báo khi phiên bản numpy/scikit-learn lúc nạp khác lúc đóng gói (pickle có thể không tương thích)."""
    from importlib import metadata

    saved = bundle.get("env") or {}
    for pkg in ("numpy", "scikit-learn"):
        old = saved.get(pkg)
        try:
            current = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
        if old and old.split(".")[:2] != current.split(".")[:2]:
            warnings.warn(
                f"Gói mô hình được đóng gói với {pkg}=={old} nhưng môi trường hiện tại là {current}; "
                "nếu gặp lỗi nạp hoặc kết quả lạ, hãy chạy lại scripts/export_model.py.",
                RuntimeWarning,
                stacklevel=2,
            )


@lru_cache(maxsize=1)
def _drop_for_what_if() -> tuple[str, ...]:
    """Đọc một lần danh sách đặc trưng không đưa vào what-if từ configs/features.yaml."""
    try:
        return tuple(load_config("features").get("drop_for_what_if", []))
    except Exception:
        return ()


class Predictor:
    """Bộ suy luận hoàn chỉnh tích hợp mô hình, chuẩn hóa, luật an toàn và giải thích."""

    def __init__(self, bundle: dict[str, Any], path: Path | str | None = None) -> None:
        """Khởi tạo Predictor từ từ điển bundle đã nạp.

        Args:
            bundle: Gói suy luận chứa mô hình, scaler, siêu dữ liệu.
            path: Đường dẫn tệp bundle nếu có.
        """
        self.bundle = bundle
        self.path = Path(path) if path is not None else None

        self.run_name: str = str(bundle.get("run_name", "unknown"))
        self.task: str = str(bundle.get("task", "multiclass"))
        self.variant: str = str(bundle.get("variant", "full"))
        self.class_names: list[str] = list(bundle.get("class_names", []))
        self.feature_order: list[str] = list(bundle.get("feature_order", list(FEATURES)))

        self.scaler_spec = bundle.get("scaler")
        if hasattr(self.scaler_spec, "transform"):
            self._scaler = self.scaler_spec
        elif isinstance(self.scaler_spec, dict):
            self._scaler = StandardScalerNP.from_dict(self.scaler_spec)
        else:
            self._scaler = None

        self.model = bundle.get("model")
        self.calibrator = bundle.get("calibrator")
        self.scale_factors = bundle.get("scale_factors")
        self.threshold = bundle.get("threshold")

        pop_mean = bundle.get("pop_mean_raw")
        if pop_mean is not None:
            self.pop_mean_raw = np.asarray(pop_mean, dtype=np.float32)
        elif self._scaler is not None and hasattr(self._scaler, "mean_") and self._scaler.mean_ is not None:
            self.pop_mean_raw = np.asarray(self._scaler.mean_, dtype=np.float32)
        else:
            self.pop_mean_raw = None

        self.risk_levels: dict[str, float] = dict(bundle.get("risk_levels") or {"low": 0.10, "medium": 0.25})
        self.what_if_cfg: dict[str, Any] = dict(bundle.get("what_if") or {})
        self.disclaimer: str = str(bundle.get("disclaimer") or DISCLAIMER_VI)

        try:
            self.rules: list[Rule] = load_rules()
        except Exception:
            self.rules = []

    @classmethod
    def from_bundle(cls, path: Path | str | None = None) -> "Predictor":
        """Nạp Predictor từ tệp gói bundle.

        Độ ưu tiên đường dẫn:
        1. Tham số `path` (nếu truyền).
        2. Biến môi trường `MODEL_PATH`.
        3. Cấu hình `configs/serve.yaml` (`model_path`).

        Args:
            path: Đường dẫn tệp bundle (tùy chọn).

        Returns:
            Predictor: Thực thể dự đoán đã sẵn sàng.

        Raises:
            ModelNotReady: Nếu tệp mô hình không tồn tại.
        """
        if path is not None:
            target_path = Path(path)
            if not target_path.is_file() and not target_path.is_absolute():
                target_path = ROOT / target_path
        else:
            env_path = get_env("MODEL_PATH")
            if env_path:
                target_path = Path(env_path)
            else:
                try:
                    cfg = load_config("serve")
                    target_path = Path(cfg.get("model_path", "models/best_model.joblib"))
                except Exception:
                    target_path = Path("models/best_model.joblib")
            if not target_path.is_absolute():
                target_path = ROOT / target_path

        if not target_path.is_file():
            raise ModelNotReady(
                "Chưa có mô hình. Hãy chạy scripts/train.py và scripts/export_model.py rồi khởi động lại API."
            )

        bundle = load_bundle(target_path)
        _warn_if_env_mismatch(bundle)
        return cls(bundle=bundle, path=target_path)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Chuẩn hóa dữ liệu đặc trưng bằng scaler."""
        X_arr = np.asarray(X, dtype=np.float32)
        if self._scaler is not None:
            return self._scaler.transform(X_arr)
        return X_arr

    def _predict_proba_scaled(self, Xs: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất và áp dụng hiệu chuẩn (nếu có)."""
        proba = self.model.predict_proba(Xs)
        proba = np.asarray(proba, dtype=np.float64)
        if self.calibrator is not None:
            if hasattr(self.calibrator, "predict"):
                proba = self.calibrator.predict(proba)
            elif hasattr(self.calibrator, "predict_proba"):
                proba = self.calibrator.predict_proba(proba)
            elif callable(self.calibrator):
                proba = self.calibrator(proba)

        proba = np.asarray(proba, dtype=np.float64)
        row_sums = proba.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return proba / row_sums

    def _predict_proba_raw(self, X_raw: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất từ ma trận đặc trưng thô."""
        X_arr = np.atleast_2d(X_raw).astype(np.float32)
        Xs = self._transform(X_arr)
        return self._predict_proba_scaled(Xs)

    def _index_fn(self, X_raw: np.ndarray) -> np.ndarray:
        """Hàm tính chỉ số nguy cơ dùng chung phục vụ giải thích và what-if (đúng định nghĩa A7).

        Args:
            X_raw: Ma trận hoặc vector đặc trưng thô chưa chuẩn hóa.

        Returns:
            np.ndarray: Mảng 1D chứa chỉ số nguy cơ của từng mẫu.
        """
        X_arr = np.atleast_2d(X_raw).astype(np.float32)
        proba = self._predict_proba_raw(X_arr)
        if self.task == "binary":
            col = 1 if proba.shape[-1] > 1 else 0
            indices = proba[:, col]
        else:
            if proba.shape[-1] >= 3:
                indices = proba[:, 2] + 0.5 * proba[:, 1]
            elif proba.shape[-1] == 2:
                indices = proba[:, 1]
            else:
                indices = proba[:, 0]
        return np.clip(indices, 0.0, 1.0).astype(np.float64)

    def predict(self, profile: dict[str, Any] | PatientProfile) -> dict[str, Any]:
        """Dự đoán nguy cơ tiểu đường từ hồ sơ sức khỏe.

        Quy trình: vector -> scaler -> mô hình -> hiệu chuẩn -> nhãn -> risk_index ->
                   evaluate(context) -> combine -> advice_for; gắn kèm disclaimer.

        Args:
            profile: Từ điển hoặc PatientProfile chứa 21 chỉ số và trường tùy chọn.

        Returns:
            dict[str, Any]: Kết quả dự đoán đầy đủ bao gồm xác suất, mức nguy cơ,
                            cảnh báo an toàn, lời khuyên và miễn trừ trách nhiệm.
        """
        if isinstance(profile, dict):
            p_obj = PatientProfile(**profile)
        elif isinstance(profile, PatientProfile):
            p_obj = profile
        else:
            raise TypeError(f"profile phải là dict hoặc PatientProfile, nhận được: {type(profile)}")

        vec = p_obj.to_vector().reshape(1, -1)
        Xs = self._transform(vec)
        proba_matrix = self._predict_proba_scaled(Xs)
        proba = proba_matrix[0]

        # Xác định nhãn dự đoán
        if self.task == "binary":
            p_pos = float(proba[1]) if len(proba) > 1 else float(proba[0])
            th = float(self.threshold) if self.threshold is not None else 0.5
            label_idx = int(p_pos >= th)
            p_diabetes = p_pos
        else:
            if self.scale_factors is not None:
                sf = np.asarray(self.scale_factors, dtype=np.float64)
                scaled = proba * sf
                label_idx = int(np.argmax(scaled))
            else:
                label_idx = int(np.argmax(proba))
            p_diabetes = float(proba[2]) if len(proba) >= 3 else float(proba[-1])

        label = self.class_names[label_idx] if label_idx < len(self.class_names) else str(label_idx)

        # Tính chỉ số nguy cơ
        r_idx = float(risk_index(proba, task=self.task))

        # Đánh giá luật an toàn
        ctx = p_obj.to_context(p_diabetes=p_diabetes, risk_index=r_idx)
        alerts = evaluate(ctx, rules=self.rules)

        # Kết hợp phân tầng nguy cơ
        low_th = float(self.risk_levels.get("low", 0.10))
        med_th = float(self.risk_levels.get("medium", 0.25))
        risk_lvl = combine(r_idx, alerts, low=low_th, medium=med_th)

        # Lời khuyên y tế (luôn chứa CONFIRM_TEST_ADVICE)
        advice = advice_for(risk_lvl, alerts=alerts)

        probabilities = {name: float(p) for name, p in zip(self.class_names, proba)}
        alerts_data = [
            {
                "rule_id": a.rule_id,
                "level": a.level,
                "text": a.text,
                "evidence": getattr(a, "evidence", {}),
            }
            for a in alerts
        ]

        return {
            "probabilities": probabilities,
            "risk_index": r_idx,
            "risk_level": risk_lvl,
            "label": label,
            "alerts": alerts_data,
            "advice": advice,
            "disclaimer": self.disclaimer,
            "model_info": self.info(),
        }

    def explain(
        self,
        profile: dict[str, Any] | PatientProfile,
        what_if_changes: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Giải thích tác động đặc trưng và giả lập kịch bản what-if lối sống.

        Args:
            profile: Từ điển hoặc PatientProfile của bệnh nhân.
            what_if_changes: Từ điển thay đổi tùy biến các đặc trưng lối sống (tùy chọn).

        Returns:
            dict[str, Any]: Kết quả giải thích gồm drivers, what_if, custom_what_if,
                            method, note và disclaimer.
        """
        if isinstance(profile, dict):
            p_obj = PatientProfile(**profile)
        elif isinstance(profile, PatientProfile):
            p_obj = profile
        else:
            raise TypeError(f"profile phải là dict hoặc PatientProfile, nhận được: {type(profile)}")

        x_raw = p_obj.to_vector()

        pop_mean = self.pop_mean_raw
        if pop_mean is None:
            if self._scaler is not None and hasattr(self._scaler, "mean_") and self._scaler.mean_ is not None:
                pop_mean = np.asarray(self._scaler.mean_, dtype=np.float32)
            else:
                pop_mean = np.zeros(len(self.feature_order), dtype=np.float32)

        drop_list = set(_drop_for_what_if())

        drivers = feature_effects(
            self._index_fn,
            x_raw,
            pop_mean,
            features=self.feature_order,
            skip=drop_list,
        )

        what_if_list = what_if_scenarios(
            self._index_fn,
            x_raw,
            self.what_if_cfg,
            features=self.feature_order,
        )

        custom_what_if = None
        if what_if_changes:
            allowed = set(self.what_if_cfg.keys())
            custom_what_if = what_if_custom(
                self._index_fn,
                x_raw,
                what_if_changes,
                allowed=allowed,
                features=self.feature_order,
            )

        return {
            "drivers": drivers,
            "what_if": what_if_list,
            "custom_what_if": custom_what_if,
            "method": "mean_replacement",
            "note": WHAT_IF_NOTE,
            "disclaimer": self.disclaimer,
        }

    def info(self) -> dict[str, Any]:
        """Trả về thông tin metadata của mô hình."""
        return {
            "run_name": self.run_name,
            "task": self.task,
            "variant": self.variant,
            "class_names": list(self.class_names),
            "feature_order": list(self.feature_order),
            "risk_levels": dict(self.risk_levels),
            "trained_on": self.bundle.get("trained_on", "BRFSS 2015 (CDC), 253.680 hồ sơ"),
            "metrics_val": self.bundle.get("metrics_val"),
            "backend": self.bundle.get("backend"),
        }
