"""Kiểm thử Predictor, luật an toàn, giải thích và miễn trừ trách nhiệm (K08c)."""

from pathlib import Path

import numpy as np
import pytest

from diabetes_ai.data.schema import FEATURES
from diabetes_ai.explain.what_if import WHAT_IF_NOTE
from diabetes_ai.inference.predictor import ModelNotReady, Predictor
from diabetes_ai.safety.disclaimers import CONFIRM_TEST_ADVICE, DISCLAIMER_VI


def test_predict_probs_sum_to_one(tiny_bundle: Path, sample_profile: dict) -> None:
    """Kiểm tra xác suất dự đoán của Predictor có tổng bằng 1 và nằm trong [0, 1]."""
    predictor = Predictor.from_bundle(tiny_bundle)
    res = predictor.predict(sample_profile)

    assert "probabilities" in res
    probs = res["probabilities"]
    assert len(probs) == len(predictor.class_names)
    for cls_name, p in probs.items():
        assert 0.0 <= p <= 1.0, f"Xác suất của {cls_name} ({p}) nằm ngoài [0, 1]"

    total = sum(probs.values())
    assert total == pytest.approx(1.0, abs=1e-6), f"Tổng xác suất ({total}) phải bằng 1"


def test_disclaimer_and_confirm_advice_at_every_level(
    tiny_bundle: Path, sample_profile: dict
) -> None:
    """Kiểm tra mọi phản hồi có miễn trừ trách nhiệm và khuyên xét nghiệm xác nhận ở mọi mức nguy cơ."""
    predictor = Predictor.from_bundle(tiny_bundle)

    # 1. Hồ sơ nguy cơ thấp
    low_profile = dict(sample_profile)
    low_profile.update({
        "HighBP": 0, "HighChol": 0, "BMI": 19.0, "GenHlth": 1,
        "Age": 1, "Smoker": 0, "HeartDiseaseorAttack": 0, "Stroke": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
    })
    res_low = predictor.predict(low_profile)
    assert res_low["risk_level"] == "thấp"

    # 2. Hồ sơ nguy cơ trung bình (nâng từ thấp nhờ cảnh báo warn R4)
    med_profile = dict(low_profile)
    med_profile["fasting_glucose"] = 110.0
    res_med = predictor.predict(med_profile)
    assert res_med["risk_level"] == "trung bình"

    # 3. Hồ sơ nguy cơ cao
    high_profile = dict(sample_profile)
    high_profile.update({
        "HighBP": 1, "HighChol": 1, "BMI": 45.0, "GenHlth": 5,
        "Age": 13, "Smoker": 1, "HeartDiseaseorAttack": 1, "Stroke": 1,
        "DiffWalk": 1, "PhysActivity": 0,
    })
    res_high = predictor.predict(high_profile)
    assert res_high["risk_level"] == "cao"

    # Kiểm tra ràng buộc an toàn ở cả 3 mức
    for res in (res_low, res_med, res_high):
        assert res["disclaimer"], "Phải luôn có tuyên bố miễn trừ trách nhiệm"
        assert "học tập" in res["disclaimer"]
        assert "chẩn đoán" in res["disclaimer"]

        advice = res["advice"]
        assert isinstance(advice, list) and len(advice) >= 1
        assert any("xét nghiệm xác nhận" in a for a in advice), (
            "Phải luôn có lời khuyên xét nghiệm xác nhận ở mọi mức nguy cơ"
        )

    # Riêng mức cao phải có lời khuyên khám sớm
    assert any("khám sớm" in a for a in res_high["advice"]), "Mức nguy cơ cao phải có lời khuyên đi khám sớm"


def test_glucose_over_250_forces_high_level(
    tiny_bundle: Path, sample_profile: dict
) -> None:
    """Kiểm tra hồ sơ nguy cơ thấp nhưng có đường huyết >= 250 bị nâng lên mức cao và có cảnh báo R1."""
    predictor = Predictor.from_bundle(tiny_bundle)

    # Hồ sơ vốn có chỉ số rất thấp
    profile = dict(sample_profile)
    profile.update({
        "HighBP": 0, "HighChol": 0, "BMI": 19.0, "GenHlth": 1,
        "Age": 1, "Smoker": 0, "HeartDiseaseorAttack": 0, "Stroke": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "glucose": 260.0,
    })

    res = predictor.predict(profile)
    assert res["risk_level"] == "cao", "Đường huyết 260 phải ép mức nguy cơ lên 'cao'"

    r1_alerts = [a for a in res["alerts"] if a["rule_id"] == "R1"]
    assert len(r1_alerts) == 1, "Phải có cảnh báo luật R1 khi glucose=260"
    assert r1_alerts[0]["level"] == "danger"
    assert "glucose" in r1_alerts[0]["evidence"]


def test_bundle_roundtrip_identical_probabilities(
    tiny_bundle: Path, sample_profile: dict
) -> None:
    """Kiểm tra nạp lại bundle cho xác suất hoàn toàn giống nhau (sai khác <= 1e-9)."""
    predictor1 = Predictor.from_bundle(tiny_bundle)
    predictor2 = Predictor.from_bundle(tiny_bundle)

    res1 = predictor1.predict(sample_profile)
    res2 = predictor2.predict(sample_profile)

    for k in res1["probabilities"]:
        diff = abs(res1["probabilities"][k] - res2["probabilities"][k])
        assert diff <= 1e-9, f"Sai khác xác suất của {k} ({diff}) vượt quá 1e-9"

    # Kiểm tra trực tiếp trên ma trận xác suất qua _predict_proba_raw
    vec = np.array([sample_profile[f] for f in FEATURES], dtype=np.float32)
    p1 = predictor1._predict_proba_raw(vec)
    p2 = predictor2._predict_proba_raw(vec)
    max_diff = float(np.max(np.abs(p1 - p2)))
    assert max_diff <= 1e-9, f"Sai khác ma trận xác suất {max_diff} vượt quá 1e-9"


def test_binary_bundle_two_probabilities_and_index_equals_p_pos(
    tiny_bundle_binary: Path, sample_profile: dict
) -> None:
    """Kiểm tra gói binary có đúng 2 xác suất và chỉ số nguy cơ r chính bằng xác suất dương tính."""
    predictor = Predictor.from_bundle(tiny_bundle_binary)
    assert predictor.task == "binary"

    res = predictor.predict(sample_profile)
    probs = res["probabilities"]

    assert len(probs) == 2, f"Gói nhị phân phải có đúng 2 lớp xác suất, nhận được {len(probs)}"
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    # Với bài toán nhị phân, risk_index bằng P(lớp dương tính)
    pos_name = predictor.class_names[1]
    p_pos = probs[pos_name]
    assert res["risk_index"] == pytest.approx(p_pos, abs=1e-6), (
        f"risk_index ({res['risk_index']}) phải bằng P(dương tính) ({p_pos})"
    )


def test_missing_bundle_raises_model_not_ready() -> None:
    """Kiểm tra khi thiếu tệp bundle phải ném ngoại lệ ModelNotReady với thông báo tiếng Việt."""
    missing_path = Path("non_existent_model_bundle_xyz.joblib")
    with pytest.raises(ModelNotReady, match="Chưa có mô hình"):
        Predictor.from_bundle(missing_path)


def test_explain_has_note_and_only_whitelisted_what_if(
    tiny_bundle: Path, sample_profile: dict
) -> None:
    """Kiểm tra giải thích có ghi chú, miễn trừ trách nhiệm và chỉ chứa đặc trưng what-if hợp lệ."""
    predictor = Predictor.from_bundle(tiny_bundle)

    profile = dict(sample_profile)
    profile["BMI"] = 32.0
    profile["Smoker"] = 1

    res = predictor.explain(profile)

    assert "note" in res
    assert res["note"] == WHAT_IF_NOTE
    assert "tương quan" in res["note"]

    assert "disclaimer" in res
    assert res["disclaimer"] == DISCLAIMER_VI

    # Kịch bản what-if chỉ được thuộc danh sách cấu hình
    whitelisted = set(predictor.what_if_cfg.keys())
    assert len(res["what_if"]) > 0
    for scenario in res["what_if"]:
        assert scenario["feature"] in whitelisted, (
            f"Đặc trưng {scenario['feature']} không nằm trong whitelist what-if: {whitelisted}"
        )

    # what_if_custom từ chối đặc trưng không thuộc whitelist (ví dụ: Age)
    with pytest.raises(ValueError, match="Chỉ được thay đổi:"):
        predictor.explain(profile, what_if_changes={"Age": 5.0})

    # what_if_custom chấp nhận đặc trưng hợp lệ
    custom_res = predictor.explain(profile, what_if_changes={"BMI": 24.5})
    assert custom_res["custom_what_if"] is not None
    assert "delta" in custom_res["custom_what_if"]
    assert "before" in custom_res["custom_what_if"]
    assert "after" in custom_res["custom_what_if"]


def test_numpy_mlp_matches_torch() -> None:
    """Kiểm tra NumpyMLP cho xác suất tương đương tuyệt đối với PyTorch (sai khác <= 1e-5)."""
    torch = pytest.importorskip("torch")
    from diabetes_ai.inference.export import NumpyMLP
    from diabetes_ai.models.mlp import MLPNet

    torch.manual_seed(42)
    net = MLPNet(n_in=21, hidden=(32, 16), n_out=3, dropout=(0.0, 0.0))
    net.eval()

    rng = np.random.default_rng(42)
    x = rng.standard_normal((10, 21)).astype(np.float32)

    with torch.no_grad():
        torch_logits = net(torch.from_numpy(x))
        torch_probs = torch.softmax(torch_logits, dim=-1).numpy().astype(np.float64)

    numpy_mlp = NumpyMLP.from_torch(net)
    numpy_probs = numpy_mlp.predict_proba(x)

    diff = float(np.max(np.abs(torch_probs - numpy_probs)))
    assert diff <= 1e-5, f"Chênh lệch tối đa giữa NumpyMLP và PyTorch là {diff} > 1e-5"
