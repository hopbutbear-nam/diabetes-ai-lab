"""Bọc hệ thống luật an toàn lâm sàng và bác sĩ phản biện (critic) thành agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diabetes_ai.inference.schemas import PatientProfile
from diabetes_ai.safety.critic import cross_check
from diabetes_ai.safety.rules import Alert, Rule, evaluate, load_rules


class SafetyAgent:
    """Tác tử an toàn kết hợp đánh giá luật lâm sàng cố định và phản biện critic."""

    def __init__(
        self,
        rules_path: Path | str | None = None,
        rules: list[Rule] | None = None,
    ) -> None:
        """Khởi tạo SafetyAgent.

        Args:
            rules_path: Đường dẫn tới file YAML quy tắc an toàn (mặc định configs/safety_rules.yaml).
            rules: Danh sách đối tượng Rule đã nạp sẵn (tùy chọn).
        """
        if rules is not None:
            self.rules = rules
        else:
            self.rules = load_rules(rules_path)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Đánh giá an toàn và kiểm tra chéo các mâu thuẫn lâm sàng.

        Args:
            state: Từ điển trạng thái chứa 'profile' và 'risk_result' từ tác tử nguy cơ.

        Returns:
            dict[str, Any]: Kết quả phân tích an toàn gồm danh sách cảnh báo và phản biện.
        """
        profile = state.get("profile", state)
        risk_result = state.get("risk_result", {})
        predictor_output = risk_result.get("raw_prediction") or {
            "risk_level": risk_result.get("risk_level", "thấp"),
            "risk_index": risk_result.get("risk_index", 0.0),
            "probabilities": risk_result.get("probabilities", {}),
        }

        # Xác định xác suất tiểu đường và chỉ số nguy cơ cho ngữ cảnh đánh giá luật
        probs = predictor_output.get("probabilities", {})
        if "Tiểu đường" in probs:
            p_diabetes = float(probs["Tiểu đường"])
        elif "1" in probs:
            p_diabetes = float(probs["1"])
        else:
            p_diabetes = float(risk_result.get("risk_index", 0.0))
        r_idx = float(predictor_output.get("risk_index", 0.0))

        # Trích xuất ngữ cảnh context phù hợp với hàm evaluate()
        profile_dict = profile if isinstance(profile, dict) else profile.model_dump()
        try:
            p_obj = PatientProfile(**profile_dict)
            context = p_obj.to_context(p_diabetes=p_diabetes, risk_index=r_idx)
        except Exception:
            symptoms_list = profile_dict.get("symptoms", [])
            context = {
                "glucose": profile_dict.get("glucose"),
                "fasting_glucose": profile_dict.get("fasting_glucose"),
                "n_symptoms": len(set(symptoms_list)) if symptoms_list else 0,
                "bmi": float(profile_dict.get("BMI", profile_dict.get("bmi", 25.0))),
                "high_bp": int(profile_dict.get("HighBP", profile_dict.get("high_bp", 0))),
                "high_chol": int(profile_dict.get("HighChol", profile_dict.get("high_chol", 0))),
                "heart_or_stroke": 1
                if (
                    profile_dict.get("HeartDiseaseorAttack", 0) == 1
                    or profile_dict.get("Stroke", 0) == 1
                )
                else 0,
                "age_code": int(profile_dict.get("Age", profile_dict.get("age_code", 1))),
                "chol_check": int(profile_dict.get("CholCheck", profile_dict.get("chol_check", 1))),
                "p_diabetes": float(p_diabetes),
                "risk_index": float(r_idx),
            }

        alerts = evaluate(context, rules=self.rules)

        critic_result = cross_check(
            profile=profile_dict,
            predictor_output=predictor_output,
            rules_alerts=alerts,
        )

        has_danger = any(getattr(a, "level", None) == "danger" for a in alerts)

        return {
            "agent": "safety_agent",
            "alerts": alerts,
            "critic": critic_result,
            "contradictions": critic_result.get("contradictions", []),
            "suggested_level": critic_result.get("suggested_level"),
            "has_danger": has_danger,
        }
