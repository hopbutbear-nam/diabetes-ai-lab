"""FastAPI: phục vụ các endpoint /health, /model-info, /predict, /explain và giao diện web."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from diabetes_ai.config import load_config
from diabetes_ai.paths import ROOT
from diabetes_ai.serving.deps import get_predictor
from diabetes_ai.serving.routes import router as api_router

logger = logging.getLogger("diabetes_ai.serving")

# Bảng tra cứu miền giá trị hợp lệ cho các trường thông tin
FIELD_RANGES: dict[str, str] = {
    "HighBP": "[0, 1]",
    "HighChol": "[0, 1]",
    "CholCheck": "[0, 1]",
    "BMI": "[10, 100]",
    "Smoker": "[0, 1]",
    "Stroke": "[0, 1]",
    "HeartDiseaseorAttack": "[0, 1]",
    "PhysActivity": "[0, 1]",
    "Fruits": "[0, 1]",
    "Veggies": "[0, 1]",
    "HvyAlcoholConsump": "[0, 1]",
    "AnyHealthcare": "[0, 1]",
    "NoDocbcCost": "[0, 1]",
    "GenHlth": "[1, 5]",
    "MentHlth": "[0, 30]",
    "PhysHlth": "[0, 30]",
    "DiffWalk": "[0, 1]",
    "Sex": "[0, 1]",
    "Age": "[1, 13]",
    "Education": "[1, 6]",
    "Income": "[1, 8]",
    "glucose": "[20, 800]",
    "fasting_glucose": "[20, 800]",
}


def _get_serve_config() -> dict[str, Any]:
    try:
        return load_config("serve")
    except Exception:
        return {}


serve_cfg = _get_serve_config()
cors_origins = serve_cfg.get(
    "cors_origins",
    [
        "http://localhost:8200",
        "http://127.0.0.1:8200",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",
    ],
)
app_dir_str = serve_cfg.get("app_dir", "app")
APP_DIR: Path = Path(app_dir_str) if Path(app_dir_str).is_absolute() else ROOT / app_dir_str
APP_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Vòng đời ứng dụng: thử nạp mô hình lúc khởi động, chỉ cảnh báo log nếu thiếu."""
    try:
        get_predictor()
        logger.info("Đã nạp mô hình thành công lúc khởi động.")
    except Exception as exc:
        logger.warning(
            "Cảnh báo: Chưa thể nạp mô hình lúc khởi động (%s). API vẫn hoạt động nhưng endpoint dự đoán sẽ trả về 503.",
            exc,
        )
    yield


app = FastAPI(
    title="Diabetes AI Lab",
    version="0.1.0",
    description="Dịch vụ phân tích sàng lọc nguy cơ tiểu đường phục vụ học tập.",
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _json_safe(value: Any) -> Any:
    """Chuyển giá trị không tuần tự hóa được (NaN/Infinity, đối tượng lạ) thành chuỗi."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Xử lý lỗi kiểm thực đầu vào, trả về mã 422 với thông điệp tiếng Việt nêu rõ trường và miền hợp lệ."""
    formatted_errors = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = str(loc[-1]) if loc else "body"
        err_type = str(err.get("type", ""))
        input_val = err.get("input")

        if err_type == "missing":
            message = f"Thiếu trường bắt buộc: {field}"
            input_val = None
        elif err_type == "extra_forbidden":
            message = f"Trường không được phép: {field}"
        elif field in FIELD_RANGES:
            message = f"{field} phải nằm trong {FIELD_RANGES[field]}"
        else:
            message = f"{field}: {err.get('msg', 'Giá trị không hợp lệ')}"

        formatted_errors.append(
            {
                "field": field,
                "message": message,
                "input": _json_safe(input_val),
            }
        )
    return JSONResponse(status_code=422, content={"detail": formatted_errors})


@app.get("/health")
def health() -> dict[str, str]:
    """Kiểm tra sức khỏe dịch vụ API (giữ nguyên không thêm trường)."""
    return {"status": "ok"}


# Gắn các route chức năng
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def root():
    """Chuyển hướng gốc tới giao diện web demo."""
    return RedirectResponse(url="/app/")


# Phục vụ tệp tĩnh giao diện web
app.mount("/app", StaticFiles(directory=str(APP_DIR), html=True), name="app")
