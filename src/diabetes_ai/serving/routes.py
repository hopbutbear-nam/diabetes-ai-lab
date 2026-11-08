"""Các endpoint phục vụ API /model-info, /predict, /explain, /consult."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from diabetes_ai.agents.orchestrator import Orchestrator
from diabetes_ai.explain.what_if import WhatIfError
from diabetes_ai.inference.predictor import Predictor
from diabetes_ai.inference.schemas import (
    ConsultResponse,
    ExplainRequest,
    ExplainResponse,
    PatientProfile,
    PredictionResponse,
)
from diabetes_ai.serving.deps import get_orchestrator, get_predictor

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
    try:
        result = predictor.explain(request.profile, what_if_changes=request.what_if_changes)
    except WhatIfError as exc:
        # Đầu vào what-if sai (khóa không được phép / ngoài miền / không hữu hạn) là lỗi của client.
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "field": "what_if_changes",
                    "message": str(exc),
                    "input": request.what_if_changes,
                }
            ],
        ) from exc
    return ExplainResponse(**result)


@router.post("/consult", response_model=ConsultResponse)
def consult(
    profile: PatientProfile,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ConsultResponse:
    """Tư vấn đa tác tử: RiskAgent → SafetyAgent (luật + phản biện) → KnowledgeAgent (trích dẫn có nguồn)."""
    return ConsultResponse(**orchestrator.consult(profile.model_dump()))
