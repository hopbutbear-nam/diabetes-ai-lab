"""Bọc mô hình dự đoán (Predictor) thành agent tính toán nguy cơ."""

from __future__ import annotations

from typing import Any

from diabetes_ai.inference.predictor import Predictor


class RiskAgent:
    """Tác tử ước lượng nguy cơ dựa trên mô hình học máy (Predictor)."""

    def __init__(self, predictor: Predictor | None = None) -> None:
        """Khởi tạo RiskAgent.

        Args:
            predictor: Đối tượng Predictor đã nạp sẵn (tùy chọn). Nếu None sẽ nạp từ bundle.
        """
        self.predictor = predictor

    def _get_predictor(self) -> Predictor:
        """Lấy hoặc khởi tạo đối tượng Predictor."""
        if self.predictor is None:
            self.predictor = Predictor.from_bundle()
        return self.predictor

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Thực thi suy luận nguy cơ từ hồ sơ trong trạng thái.

        Args:
            state: Từ điển trạng thái chứa trường 'profile' (hoặc trực tiếp hồ sơ người dùng).

        Returns:
            dict[str, Any]: Kết quả dự đoán có cấu trúc gồm xác suất, chỉ số và mức nguy cơ.
        """
        profile = state.get("profile", state)
        predictor = self._get_predictor()
        prediction = predictor.predict(profile)

        return {
            "agent": "risk_agent",
            "probabilities": prediction.get("probabilities", {}),
            "risk_index": float(prediction.get("risk_index", 0.0)),
            "risk_level": str(prediction.get("risk_level", "thấp")),
            "label": str(prediction.get("label", "")),
            "raw_prediction": prediction,
        }
