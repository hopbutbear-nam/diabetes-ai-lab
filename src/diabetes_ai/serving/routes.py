"""Các endpoint phục vụ API /model-info, /predict, /explain."""

from typing import Any
from fastapi import APIRouter, Depends

from diabetes_ai.inference.predictor import Predictor
from diabetes_ai.inference.schemas import (
    ExplainRequest,
    ExplainResponse,
    PatientProfile,
    PredictionResponse,
)
from diabetes_ai.serving.deps import get_predictor

router = APIRouter()


@router.get("/model-info")
def get_model_info(predictor: Predictor = Depends(get_predictor)) -> dict[str, Any]:
    """Trả về thông tin metadata của mô hình đang phục vụ."""
    return predictor.info()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    profile: PatientProfile,
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    """Dự đoán nguy cơ tiểu đường từ thông tin hồ sơ sức khỏe."""
    result = predictor.predict(profile)
    return PredictionResponse(**result)


@router.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    predictor: Predictor = Depends(get_predictor),
) -> ExplainResponse:
    """Giải thích các yếu tố rủi ro và mô phỏng kịch bản what-if."""
    result = predictor.explain(request.profile, what_if_changes=request.what_if_changes)
    return ExplainResponse(**result)
