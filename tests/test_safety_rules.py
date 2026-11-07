"""Kiểm thử luật an toàn, phân tầng nguy cơ và miễn trừ trách nhiệm (K08a)."""

import pytest

from diabetes_ai.safety.disclaimers import (
    CONFIRM_TEST_ADVICE,
    DISCLAIMER_SHORT,
    DISCLAIMER_VI,
    EARLY_EXAM_ADVICE,
    advice_for,
)
from diabetes_ai.safety.risk_level import combine, level_from_index, risk_index
from diabetes_ai.safety.rules import Alert, Rule, evaluate, load_rules, top_level


def test_loads_ten_rules_unique_ids():
    """Kiểm tra nạp đúng 10 luật từ cấu hình YAML với ID duy nhất và mức hợp lệ."""
    rules = load_rules()
    assert len(rules) == 10, f"Kỳ vọng đúng 10 luật, thực tế có {len(rules)}"

    ids = [r.id for r in rules]
    assert len(ids) == len(set(ids)), f"Các ID luật phải là duy nhất: {ids}"

    valid_levels = {"danger", "warn", "info"}
    for r in rules:
        assert r.level in valid_levels, f"Mức cảnh báo '{r.level}' của luật '{r.id}' không hợp lệ"
        assert len(r.when.strip()) > 0, f"Luật '{r.id}' thiếu điều kiện 'when'"
        assert len(r.text.strip()) > 0, f"Luật '{r.id}' thiếu mô tả 'text'"

    expected_ids = {f"R{i}" for i in range(1, 11)}
    assert set(ids) == expected_ids, f"Danh sách luật phải chứa đầy đủ từ R1 đến R10: {set(ids)}"


def test_r1_danger_at_250():
    """Kiểm tra luật R1: glucose >= 250 cảnh báo danger, dưới 250 không kích hoạt."""
    alerts_250 = evaluate({"glucose": 250})
    r1_alerts = [a for a in alerts_250 if a.rule_id == "R1"]
    assert len(r1_alerts) == 1, "R1 phải kích hoạt khi glucose == 250"
    assert r1_alerts[0].level == "danger"
    assert "glucose" in r1_alerts[0].evidence

    alerts_249 = evaluate({"glucose": 249.9})
    assert not any(a.rule_id == "R1" for a in alerts_249), "R1 không được kích hoạt khi glucose < 250"


def test_r2_glucose_200_with_symptom():
    """Kiểm tra luật R2: glucose >= 200 và n_symptoms >= 1."""
    alerts_hit = evaluate({"glucose": 200, "n_symptoms": 1})
    r2_hits = [a for a in alerts_hit if a.rule_id == "R2"]
    assert len(r2_hits) == 1, "R2 phải kích hoạt khi glucose >= 200 và n_symptoms >= 1"
    assert r2_hits[0].level == "danger"

    alerts_no_symptom = evaluate({"glucose": 200, "n_symptoms": 0})
    assert not any(a.rule_id == "R2" for a in alerts_no_symptom), "R2 không được kích hoạt nếu n_symptoms == 0"

    alerts_low_glucose = evaluate({"glucose": 199, "n_symptoms": 2})
    assert not any(a.rule_id == "R2" for a in alerts_low_glucose), "R2 không được kích hoạt nếu glucose < 200"


def test_missing_glucose_skips_glucose_rules():
    """Kiểm tra giá trị glucose là None khiến các luật liên quan đường huyết không chạy."""
    context = {"glucose": None, "fasting_glucose": None, "bmi": 22.0}
    alerts = evaluate(context)

    glucose_rule_ids = {"R1", "R2", "R3", "R4", "R5"}
    triggered_ids = {a.rule_id for a in alerts}
    assert not (glucose_rule_ids & triggered_ids), (
        f"Không được kích hoạt luật đường huyết khi giá trị là None: {glucose_rule_ids & triggered_ids}"
    )


def test_fasting_glucose_r3_r4_boundaries():
    """Kiểm tra ngưỡng đường huyết đói: 126 -> R3; 110 -> R4; 99 -> không."""
    # Mức 126 -> R3 (>= 126)
    alerts_126 = evaluate({"fasting_glucose": 126})
    r3_hits = [a for a in alerts_126 if a.rule_id == "R3"]
    assert len(r3_hits) == 1, "fasting_glucose = 126 phải kích hoạt R3"
    assert not any(a.rule_id == "R4" for a in alerts_126), "fasting_glucose = 126 không được kích hoạt R4"

    # Mức 110 -> R4 (100 <= fasting_glucose <= 125)
    alerts_110 = evaluate({"fasting_glucose": 110})
    r4_hits = [a for a in alerts_110 if a.rule_id == "R4"]
    assert len(r4_hits) == 1, "fasting_glucose = 110 phải kích hoạt R4"
    assert not any(a.rule_id == "R3" for a in alerts_110), "fasting_glucose = 110 không được kích hoạt R3"

    # Mức 99 -> không kích hoạt cả R3 và R4
    alerts_99 = evaluate({"fasting_glucose": 99})
    assert not any(a.rule_id in {"R3", "R4"} for a in alerts_99), "fasting_glucose = 99 không kích hoạt R3 hoặc R4"


def test_r5_low_glucose_danger():
    """Kiểm tra luật R5: glucose < 70 cảnh báo nguy cơ hạ đường huyết nguy hiểm."""
    alerts_69 = evaluate({"glucose": 69.9})
    r5_hits = [a for a in alerts_69 if a.rule_id == "R5"]
    assert len(r5_hits) == 1, "glucose < 70 phải kích hoạt R5"
    assert r5_hits[0].level == "danger"

    alerts_70 = evaluate({"glucose": 70.0})
    assert not any(a.rule_id == "R5" for a in alerts_70), "glucose == 70 không được kích hoạt R5"


def test_r7_bmi_boundary():
    """Kiểm tra ranh giới BMI luật R7: 30 có kích hoạt, 29.9 không kích hoạt."""
    alerts_30 = evaluate({"bmi": 30.0})
    r7_hits = [a for a in alerts_30 if a.rule_id == "R7"]
    assert len(r7_hits) == 1, "bmi = 30.0 phải kích hoạt R7 (warn)"
    assert r7_hits[0].level == "warn"

    alerts_29_9 = evaluate({"bmi": 29.9})
    assert not any(a.rule_id == "R7" for a in alerts_29_9), "bmi = 29.9 không được kích hoạt R7"


def test_combine_levels():
    """Kiểm tra kết hợp mức nguy cơ mô hình và cảnh báo: danger nâng thấp -> cao; warn nâng thấp -> trung bình."""
    # low risk index (0.05 -> mức thấp ban đầu)
    danger_alert = Alert(rule_id="R1", level="danger", text="Đường huyết rất cao", evidence={"glucose": 260})
    warn_alert = Alert(rule_id="R7", level="warn", text="Béo phì", evidence={"bmi": 32.0})
    info_alert = Alert(rule_id="R9", level="info", text="Mỡ máu", evidence={"age_code": 7})

    # danger nâng mức thấp -> cao
    assert combine(0.05, [danger_alert]) == "cao"

    # warn nâng mức thấp -> trung bình
    assert combine(0.05, [warn_alert]) == "trung bình"

    # info giữ nguyên mức thấp
    assert combine(0.05, [info_alert]) == "thấp"

    # không có cảnh báo giữ nguyên mức thấp
    assert combine(0.05, []) == "thấp"

    # danger nâng mức trung bình -> cao
    assert combine(0.15, [danger_alert]) == "cao"

    # warn giữ nguyên mức trung bình
    assert combine(0.15, [warn_alert]) == "trung bình"

    # mức cao sẵn thì giữ cao
    assert combine(0.30, [warn_alert]) == "cao"

    # Kiểm tra top_level
    assert top_level([info_alert, danger_alert, warn_alert]) == "danger"
    assert top_level([info_alert, warn_alert]) == "warn"
    assert top_level([info_alert]) == "info"
    assert top_level([]) is None

    # Kiểm tra lời khuyên đi kèm advice_for
    adv_low = advice_for("thấp", [])
    assert CONFIRM_TEST_ADVICE in adv_low
    assert not any("khám sớm" in a for a in adv_low)

    adv_high = advice_for("cao", [])
    assert CONFIRM_TEST_ADVICE in adv_high
    assert any("khám sớm" in a for a in adv_high)

    adv_danger = advice_for("thấp", [danger_alert])
    assert any("khám sớm" in a for a in adv_danger)


def test_level_from_index_boundaries():
    """Kiểm tra các ranh giới chỉ số nguy cơ: 0.0999 thấp; 0.10 và 0.2499 trung bình; 0.25 cao."""
    assert level_from_index(0.0999) == "thấp"
    assert level_from_index(0.10) == "trung bình"
    assert level_from_index(0.2499) == "trung bình"
    assert level_from_index(0.25) == "cao"

    # Kiểm tra thêm hàm risk_index
    # Đa lớp: P2 + 0.5 * P1
    assert pytest.approx(risk_index([0.7, 0.2, 0.1], task="multiclass")) == 0.1 + 0.5 * 0.2  # 0.20
    # Nhị phân: P(dương)
    assert pytest.approx(risk_index([0.8, 0.2], task="binary")) == 0.2


def test_unsafe_expression_rejected(tmp_path):
    """Kiểm tra từ chối biểu thức không an toàn qua AST và ném lỗi khi nạp hoặc đánh giá."""
    # Biểu thức chứa lời gọi hàm bị từ chối
    unsafe_rule_call = Rule(id="RX", level="warn", when="__import__('os').system('ls')", text="Nguy hiểm")
    with pytest.raises(ValueError, match="Biểu thức luật không an toàn"):
        evaluate({"x": 1}, [unsafe_rule_call])

    # Biểu thức chứa thuộc tính bị từ chối
    unsafe_rule_attr = Rule(id="RY", level="danger", when="x.__class__ == int", text="Nguy hiểm")
    with pytest.raises(ValueError, match="Biểu thức luật không an toàn"):
        evaluate({"x": 1}, [unsafe_rule_attr])

    # File YAML chứa ID trùng lặp
    dup_yaml = tmp_path / "dup_rules.yaml"
    dup_yaml.write_text(
        """rules:\n  - { id: R1, level: warn, when: "bmi >= 30", text: "T1" }\n  - { id: R1, level: danger, when: "bmi >= 35", text: "T2" }\n""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ID luật bị trùng"):
        load_rules(dup_yaml)
