"""Văn bản miễn trừ trách nhiệm và hướng dẫn xét nghiệm đi kèm kết quả sàng lọc."""

from __future__ import annotations

from typing import Any

from diabetes_ai.safety.rules import Alert

DISCLAIMER_VI = (
    "Chỉ để học tập và nghiên cứu, không phải chẩn đoán y khoa. "
    "Kết quả từ hệ thống chỉ mang tính chất sàng lọc nguy cơ ban đầu và không thay thế cho "
    "tư vấn, chẩn đoán hay điều trị y tế chuyên nghiệp của bác sĩ."
)

DISCLAIMER_SHORT = "Chỉ để học tập, không phải chẩn đoán y khoa."

CONFIRM_TEST_ADVICE = (
    "Kết quả chỉ là sàng lọc nguy cơ; hãy xét nghiệm xác nhận (đường huyết đói hoặc HbA1c) tại cơ sở y tế."
)

EARLY_EXAM_ADVICE = "Nên đi khám sớm tại cơ sở y tế."


def advice_for(level: str, alerts: list[Alert | Any] | None = None) -> list[str]:
    """Tạo danh sách lời khuyên dựa trên mức nguy cơ tổng hợp và danh sách cảnh báo.

    Quy định:
    - Luôn chứa CONFIRM_TEST_ADVICE ở mọi mức nguy cơ.
    - Nếu mức nguy cơ là 'cao' hoặc có cảnh báo mức 'danger', thêm lời khuyên nên đi khám sớm.

    Args:
        level: Mức nguy cơ ('thấp', 'trung bình', 'cao').
        alerts: Danh sách cảnh báo an toàn đã kích hoạt (nếu có).

    Returns:
        list[str]: Danh sách các lời khuyên y tế.
    """
    advice: list[str] = [CONFIRM_TEST_ADVICE]
    has_danger = bool(alerts and any(getattr(a, "level", None) == "danger" for a in alerts))

    if level == "cao" or has_danger:
        advice.append(EARLY_EXAM_ADVICE)

    return advice
