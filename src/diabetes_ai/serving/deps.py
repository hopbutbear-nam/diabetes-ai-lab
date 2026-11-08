"""Quản lý phụ thuộc nạp mô hình Predictor (singleton lười)."""

from pathlib import Path

from fastapi import HTTPException, status

from diabetes_ai.config import get_env, load_config
from diabetes_ai.inference.predictor import ModelNotReady, Predictor
from diabetes_ai.paths import ROOT

_PREDICTOR: Predictor | None = None


def reset_predictor() -> None:
    """Xóa bỏ instance Predictor hiện tại để buộc nạp lại (dùng trong kiểm thử)."""
    global _PREDICTOR
    _PREDICTOR = None


def get_predictor() -> Predictor:
    """Lấy thể hiện singleton Predictor.

    Độ ưu tiên đường dẫn gói:
    1. Biến môi trường MODEL_PATH.
    2. Cấu hình configs/serve.yaml (trường model_path).
    3. Mặc định models/best_model.joblib.

    Raises:
        HTTPException(503): Khi tệp mô hình chưa sẵn sàng hoặc gặp lỗi nạp.
    """
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    model_path = get_env("MODEL_PATH")
    if not model_path:
        try:
            cfg = load_config("serve")
            model_path = cfg.get("model_path", "models/best_model.joblib")
        except Exception:
            model_path = "models/best_model.joblib"

    target_path = Path(model_path)
    if not target_path.is_absolute():
        target_path = ROOT / target_path

    try:
        _PREDICTOR = Predictor.from_bundle(target_path)
        return _PREDICTOR
    except ModelNotReady as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": f"Không thể nạp mô hình: {exc}",
            },
        ) from exc
