"""Mô-đun Critic (bác sĩ phản biện) kiểm tra chéo mô hình với số đo thực tế và tính nhất quán.

Thực hiện 4 phép kiểm tra C1–C4 theo đặc tả thiết kế:
- C1: Đường huyết đói >= 126 mg/dL mà chỉ số nguy cơ chưa 'cao', hoặc 100-125 mg/dL mà mức 'thấp'.
- C2: Có từ 2 triệu chứng điển hình trở lên mà mức nguy cơ là 'thấp'.
- C3: Khoảng cách giữa 2 xác suất cao nhất nhỏ hơn 10 điểm phần trăm (< 0.10).
- C4: Thay đổi BMI +/- 2 (khoảng +/- 5 kg) làm mức nguy cơ thay đổi.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

LEVEL_RANK: dict[str, int] = {
    "thấp": 0,
    "trung bình": 1,
    "cao": 2,
}


class CheckResult(dict):
    """Kết quả chi tiết của từng phép kiểm tra từ critic."""

    def __init__(self, id: str, passed: bool, evidence: Any) -> None:
        super().__init__(id=id, passed=passed, evidence=evidence)
        self.id = id
        self.passed = passed
        self.evidence = evidence


def _extract_risk_level(output: dict[str, Any] | str | None) -> str:
    """Trích xuất mức nguy cơ dạng chuỗi từ kết quả dự đoán."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if "risk_level" in output and output["risk_level"]:
            return str(output["risk_level"])
        if "level" in output and output["level"]:
            return str(output["level"])
        if "risk_index" in output and output["risk_index"] is not None:
            r = float(output["risk_index"])
            if r < 0.10:
                return "thấp"
            if r < 0.25:
                return "trung bình"
            return "cao"
    return "thấp"


def cross_check(
    profile: dict[str, Any],
    predictor_output: dict[str, Any],
    rules_alerts: list[Any] | None = None,
    predict_fn: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Kiểm tra chéo kết quả dự đoán với thông tin lâm sàng và kiểm tra độ nhạy mô hình.

    Args:
        profile: Hồ sơ lâm sàng người dùng (chứa fasting_glucose, symptoms, BMI, ...).
        predictor_output: Kết quả từ Predictor (chứa risk_level, risk_index, probabilities).
        rules_alerts: Danh sách các cảnh báo luật an toàn kích hoạt (tùy chọn).
        predict_fn: Hàm suy luận chạy lại mô hình với dữ liệu biến thiên (tùy chọn).

    Returns:
        dict[str, Any]: Gồm 'contradictions', 'checks', và 'suggested_level'.
    """
    checks: list[CheckResult] = []
    current_level = _extract_risk_level(predictor_output)
    suggested_level = current_level

    def elevate(target: str) -> None:
        nonlocal suggested_level
        if LEVEL_RANK.get(target, 0) > LEVEL_RANK.get(suggested_level, 0):
            suggested_level = target

    # -------------------------------------------------------------------------
    # C1: Đường huyết đói
    # -------------------------------------------------------------------------
    fasting_glucose = profile.get("fasting_glucose")
    if fasting_glucose is not None:
        fg = float(fasting_glucose)
        if fg >= 126.0:
            if current_level != "cao":
                passed = False
                evidence = (
                    f"Đường huyết đói {fg} mg/dL (ngưỡng tiểu đường ADA >= 126) "
                    f"nhưng mô hình báo mức '{current_level}'"
                )
                elevate("cao")
            else:
                passed = True
                evidence = f"Đường huyết đói {fg} mg/dL phù hợp với mức 'cao'"
            checks.append(CheckResult(id="C1", passed=passed, evidence=evidence))
        elif 100.0 <= fg <= 125.0:
            if current_level == "thấp":
                passed = False
                evidence = (
                    f"Đường huyết đói {fg} mg/dL (vùng tiền tiểu đường 100-125) "
                    f"nhưng mô hình báo mức 'thấp'"
                )
                elevate("trung bình")
            else:
                passed = True
                evidence = f"Đường huyết đói {fg} mg/dL phù hợp với mức '{current_level}'"
            checks.append(CheckResult(id="C1", passed=passed, evidence=evidence))
        else:
            # Dưới 100 mg/dL: bình thường
            passed = True
            evidence = f"Đường huyết đói {fg} mg/dL ở mức bình thường (< 100 mg/dL)"
            checks.append(CheckResult(id="C1", passed=passed, evidence=evidence))

    # -------------------------------------------------------------------------
    # C2: Triệu chứng điển hình
    # -------------------------------------------------------------------------
    n_symptoms: int | None = None
    if "symptoms" in profile and profile["symptoms"] is not None:
        sym = profile["symptoms"]
        n_symptoms = len(sym) if isinstance(sym, (list, tuple, set)) else int(sym)
    elif "n_symptoms" in profile and profile["n_symptoms"] is not None:
        n_symptoms = int(profile["n_symptoms"])

    if n_symptoms is not None:
        if n_symptoms >= 2:
            if current_level == "thấp":
                passed = False
                evidence = (
                    f"Người dùng có {n_symptoms} triệu chứng điển hình "
                    f"nhưng mô hình báo mức 'thấp'"
                )
                elevate("trung bình")
            else:
                passed = True
                evidence = f"Số triệu chứng ({n_symptoms}) phù hợp với mức '{current_level}'"
            checks.append(CheckResult(id="C2", passed=passed, evidence=evidence))
        else:
            passed = True
            evidence = f"Số triệu chứng ({n_symptoms}) dưới ngưỡng cảnh báo mâu thuẫn"
            checks.append(CheckResult(id="C2", passed=passed, evidence=evidence))

    # -------------------------------------------------------------------------
    # C3: Khoảng cách giữa 2 xác suất cao nhất < 10 điểm %
    # -------------------------------------------------------------------------
    probs_raw = predictor_output.get("probabilities")
    if probs_raw is not None:
        if isinstance(probs_raw, dict):
            prob_list = [float(v) for v in probs_raw.values()]
        elif isinstance(probs_raw, (list, np.ndarray)):
            prob_list = [float(v) for v in probs_raw]
        else:
            prob_list = []

        if len(prob_list) >= 2:
            sorted_probs = sorted(prob_list, reverse=True)
            diff = sorted_probs[0] - sorted_probs[1]
            if diff < 0.10:
                passed = False
                evidence = (
                    f"Hai nhóm xác suất đứng đầu ({sorted_probs[0]:.3f} và {sorted_probs[1]:.3f}) "
                    f"cách nhau {diff * 100:.1f} điểm % (< 10 điểm %)"
                )
            else:
                passed = True
                evidence = (
                    f"Hai nhóm xác suất đứng đầu ({sorted_probs[0]:.3f} và {sorted_probs[1]:.3f}) "
                    f"cách nhau {diff * 100:.1f} điểm % (>= 10 điểm %)"
                )
            checks.append(CheckResult(id="C3", passed=passed, evidence=evidence))

    # -------------------------------------------------------------------------
    # C4: Nhạy cảm khi đổi BMI +/- 2
    # -------------------------------------------------------------------------
    bmi_val = profile.get("BMI", profile.get("bmi"))
    if predict_fn is not None and callable(predict_fn) and bmi_val is not None:
        try:
            bmi_float = float(bmi_val)
            p_plus = dict(profile)
            p_minus = dict(profile)
            new_bmi_plus = bmi_float + 2.0
            new_bmi_minus = max(10.0, bmi_float - 2.0)

            if "BMI" in p_plus:
                p_plus["BMI"] = new_bmi_plus
                p_minus["BMI"] = new_bmi_minus
            else:
                p_plus["bmi"] = new_bmi_plus
                p_minus["bmi"] = new_bmi_minus

            out_plus = predict_fn(p_plus)
            out_minus = predict_fn(p_minus)

            level_plus = _extract_risk_level(out_plus)
            level_minus = _extract_risk_level(out_minus)

            if level_plus != current_level or level_minus != current_level:
                passed = False
                evidence = (
                    f"Mức nguy cơ bị đổi khi BMI +/- 2: gốc '{current_level}', "
                    f"BMI+2 -> '{level_plus}', BMI-2 -> '{level_minus}'"
                )
            else:
                passed = True
                evidence = (
                    f"Mức nguy cơ ổn định '{current_level}' khi thay đổi BMI +/- 2 "
                    f"({new_bmi_minus:.1f} đến {new_bmi_plus:.1f})"
                )
            checks.append(CheckResult(id="C4", passed=passed, evidence=evidence))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                CheckResult(
                    id="C4",
                    passed=True,
                    evidence=f"Không thể kiểm tra C4 do lỗi gọi predict_fn: {exc}",
                )
            )

    # -------------------------------------------------------------------------
    # Kết hợp các cảnh báo an toàn từ luật (nếu có)
    # -------------------------------------------------------------------------
    if rules_alerts:
        has_danger = any(
            getattr(a, "level", None) == "danger"
            or (isinstance(a, dict) and a.get("level") == "danger")
            for a in rules_alerts
        )
        has_warn = any(
            getattr(a, "level", None) == "warn"
            or (isinstance(a, dict) and a.get("level") == "warn")
            for a in rules_alerts
        )
        if has_danger:
            elevate("cao")
        elif has_warn:
            elevate("trung bình")

    contradictions = [c.id for c in checks if not c.passed]

    return {
        "contradictions": contradictions,
        "checks": checks,
        "suggested_level": suggested_level,
    }
