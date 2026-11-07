"""Kiểm thử mô-đun bác sĩ phản biện (critic) với 4 phép thử C1–C4."""

from diabetes_ai.safety.critic import cross_check


def test_c1_flags_fasting_glucose_190_with_low_risk() -> None:
    """C1: Đường huyết đói >= 126 mg/dL nhưng mô hình dự đoán mức thấp -> cảnh báo mâu thuẫn."""
    profile = {"fasting_glucose": 190.0}
    predictor_output = {
        "risk_level": "thấp",
        "risk_index": 0.05,
        "probabilities": {"Không": 0.90, "Tiền tiểu đường": 0.06, "Tiểu đường": 0.04},
    }
    result = cross_check(profile=profile, predictor_output=predictor_output)
    assert "C1" in result["contradictions"]
    c1 = next(c for c in result["checks"] if c["id"] == "C1")
    assert not c1["passed"]
    assert result["suggested_level"] == "cao"


def test_c2_two_symptoms_low_risk() -> None:
    """C2: Có từ 2 triệu chứng điển hình nhưng mô hình dự đoán mức thấp -> nâng lên trung bình."""
    profile = {"symptoms": ["khát nước nhiều", "đi tiểu nhiều"]}
    predictor_output = {
        "risk_level": "thấp",
        "risk_index": 0.07,
        "probabilities": {"Không": 0.88, "Tiền tiểu đường": 0.08, "Tiểu đường": 0.04},
    }
    result = cross_check(profile=profile, predictor_output=predictor_output)
    assert "C2" in result["contradictions"]
    c2 = next(c for c in result["checks"] if c["id"] == "C2")
    assert not c2["passed"]
    assert result["suggested_level"] == "trung bình"


def test_c3_close_top_two() -> None:
    """C3: Hai xác suất cao nhất cách nhau dưới 10 điểm % (< 0.10) -> cảnh báo độ phân định hẹp."""
    profile = {}
    predictor_output = {
        "risk_level": "trung bình",
        "risk_index": 0.35,
        "probabilities": {"Không": 0.45, "Tiền tiểu đường": 0.42, "Tiểu đường": 0.13},
    }
    result = cross_check(profile=profile, predictor_output=predictor_output)
    assert "C3" in result["contradictions"]
    c3 = next(c for c in result["checks"] if c["id"] == "C3")
    assert not c3["passed"]


def test_no_data_no_check() -> None:
    """Chỉ chạy các phép thử có dữ liệu tương ứng; không tự bịa số đo đường huyết hay triệu chứng."""
    profile = {"BMI": 22.0}
    predictor_output = {
        "risk_level": "thấp",
        "risk_index": 0.05,
        "probabilities": {"Không": 0.90, "Tiền tiểu đường": 0.06, "Tiểu đường": 0.04},
    }
    result = cross_check(profile=profile, predictor_output=predictor_output)
    check_ids = [c["id"] for c in result["checks"]]
    assert "C1" not in check_ids
    assert "C2" not in check_ids
    assert "C4" not in check_ids
    assert len(result["contradictions"]) == 0
    assert result["suggested_level"] == "thấp"
