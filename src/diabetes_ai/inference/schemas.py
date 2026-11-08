"""Lược đồ Pydantic cho dữ liệu đầu vào và đầu ra của suy luận (inference)."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from diabetes_ai.data.schema import FEATURES

SymptomType = Literal[
    "khát nước nhiều",
    "đi tiểu nhiều",
    "mệt mỏi",
    "nhìn mờ",
    "vết thương lâu lành",
    "sụt cân không rõ lý do",
]


class PatientProfile(BaseModel):
    """Hồ sơ sức khỏe người dùng phục vụ sàng lọc nguy cơ tiểu đường."""

    model_config = ConfigDict(extra="forbid")

    # 14 đặc trưng nhị phân (0 hoặc 1)
    HighBP: int = Field(ge=0, le=1, description="Huyết áp cao (0: không, 1: có)")
    HighChol: int = Field(ge=0, le=1, description="Mỡ máu cao (0: không, 1: có)")
    CholCheck: int = Field(ge=0, le=1, description="Kiểm tra mỡ máu trong 5 năm (0: không, 1: có)")
    BMI: float = Field(ge=10.0, le=100.0, description="Chỉ số khối cơ thể (10 - 100)")
    Smoker: int = Field(ge=0, le=1, description="Hút ít nhất 100 điếu thuốc trong đời (0: không, 1: có)")
    Stroke: int = Field(ge=0, le=1, description="Từng bị đột quỵ (0: không, 1: có)")
    HeartDiseaseorAttack: int = Field(ge=0, le=1, description="Bệnh tim mạch vành hoặc nhồi máu cơ tim (0: không, 1: có)")
    PhysActivity: int = Field(ge=0, le=1, description="Hoạt động thể chất trong 30 ngày qua (0: không, 1: có)")
    Fruits: int = Field(ge=0, le=1, description="Ăn trái cây >= 1 lần/ngày (0: không, 1: có)")
    Veggies: int = Field(ge=0, le=1, description="Ăn rau >= 1 lần/ngày (0: không, 1: có)")
    HvyAlcoholConsump: int = Field(ge=0, le=1, description="Uống nhiều rượu bia (0: không, 1: có)")
    AnyHealthcare: int = Field(ge=0, le=1, description="Có bảo hiểm y tế (0: không, 1: có)")
    NoDocbcCost: int = Field(ge=0, le=1, description="Không thể đi khám vì chi phí trong 12 tháng (0: không, 1: có)")
    GenHlth: int = Field(ge=1, le=5, description="Sức khỏe tổng quát (1: xuất sắc đến 5: kém)")
    MentHlth: int = Field(ge=0, le=30, description="Số ngày sức khỏe tinh thần không tốt (0 - 30)")
    PhysHlth: int = Field(ge=0, le=30, description="Số ngày sức khỏe thể chất không tốt (0 - 30)")
    DiffWalk: int = Field(ge=0, le=1, description="Khó khăn nghiêm trọng khi đi bộ/leo cầu thang (0: không, 1: có)")
    Sex: int = Field(ge=0, le=1, description="Giới tính (0: nữ, 1: nam)")
    Age: int = Field(ge=1, le=13, description="Nhóm tuổi (1: 18-24 đến 13: >= 80)")
    Education: int = Field(ge=1, le=6, description="Trình độ học vấn (1: chưa tiểu học đến 6: đại học trở lên)")
    Income: int = Field(ge=1, le=8, description="Mức thu nhập (1: < 10k$ đến 8: >= 75k$)")

    # Trường tùy chọn không đưa vào mô hình ML, dùng cho luật an toàn
    glucose: float | None = Field(default=None, ge=20.0, le=800.0, description="Đường huyết ngẫu nhiên (mg/dL)")
    fasting_glucose: float | None = Field(default=None, ge=20.0, le=800.0, description="Đường huyết lúc đói (mg/dL)")
    symptoms: list[SymptomType] = Field(default_factory=list, description="Danh sách triệu chứng điển hình")

    def to_vector(self) -> np.ndarray:
        """Chuyển đổi 21 đặc trưng thành vector numpy float32 theo đúng thứ tự FEATURES."""
        return np.array([float(getattr(self, feat)) for feat in FEATURES], dtype=np.float32)

    def to_context(self, p_diabetes: float, risk_index: float) -> dict[str, Any]:
        """Tạo từ điển ngữ cảnh cho việc đánh giá các luật an toàn.

        Args:
            p_diabetes: Xác suất dự đoán mắc tiểu đường từ mô hình.
            risk_index: Chỉ số nguy cơ tổng hợp từ mô hình.

        Returns:
            dict[str, Any]: Ngữ cảnh chứa các biến đầu vào cho luật.
        """
        return {
            "glucose": self.glucose,
            "fasting_glucose": self.fasting_glucose,
            "n_symptoms": len(set(self.symptoms)),
            "bmi": float(self.BMI),
            "high_bp": int(self.HighBP),
            "high_chol": int(self.HighChol),
            "heart_or_stroke": 1 if (self.HeartDiseaseorAttack == 1 or self.Stroke == 1) else 0,
            "age_code": int(self.Age),
            "chol_check": int(self.CholCheck),
            "p_diabetes": float(p_diabetes),
            "risk_index": float(risk_index),
        }


class Alert(BaseModel):
    """Lược đồ cảnh báo an toàn được kích hoạt khi thỏa mãn luật."""

    model_config = ConfigDict(from_attributes=True)

    rule_id: str
    level: str
    text: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Lược đồ phản hồi dự đoán nguy cơ tiểu đường."""

    probabilities: dict[str, float]
    risk_index: float
    risk_level: str
    label: str
    alerts: list[Alert] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    disclaimer: str
    model_info: dict[str, Any] = Field(default_factory=dict)


class ExplainRequest(BaseModel):
    """Lược đồ yêu cầu giải thích nguy cơ và kịch bản what-if."""

    profile: PatientProfile
    what_if_changes: dict[str, float] | None = None

    @field_validator("what_if_changes")
    @classmethod
    def _check_finite(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        """Từ chối NaN/Infinity trong các giá trị what-if ngay ở lớp lược đồ."""
        if value is None:
            return value
        for key, val in value.items():
            if not math.isfinite(val):
                raise ValueError(f"Giá trị của {key} phải là số hữu hạn.")
        return value


class ExplainResponse(BaseModel):
    """Lược đồ phản hồi giải thích nguy cơ và kịch bản what-if."""

    drivers: list[dict[str, Any]]
    what_if: list[dict[str, Any]]
    custom_what_if: dict[str, Any] | None = None
    method: str
    note: str
    disclaimer: str
