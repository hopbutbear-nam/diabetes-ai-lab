"""Kiểm thử API phục vụ dự đoán, giải thích, kiểm tra lỗi và CORS."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from diabetes_ai.serving.app import app
from diabetes_ai.serving.deps import reset_predictor


@pytest.fixture(autouse=True)
def setup_predictor(monkeypatch: pytest.MonkeyPatch, tiny_bundle: Path):
    """Thiết lập biến môi trường MODEL_PATH và làm mới Predictor trước mỗi bài test."""
    monkeypatch.setenv("MODEL_PATH", str(tiny_bundle))
    reset_predictor()
    yield
    reset_predictor()


@pytest.fixture
def client() -> TestClient:
    """Fixture trả về TestClient cho ứng dụng FastAPI."""
    return TestClient(app)


def test_health_unchanged(client: TestClient) -> None:
    """Kiểm tra endpoint /health trả đúng 200 và {'status': 'ok'} không đổi."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_ok_has_required_keys_and_disclaimer(
    client: TestClient, sample_profile: dict
) -> None:
    """Kiểm tra /predict thành công: tổng xác suất bằng 1, có disclaimer và lời khuyên xét nghiệm xác nhận."""
    res = client.post("/predict", json=sample_profile)
    assert res.status_code == 200
    data = res.json()

    assert "probabilities" in data
    probs = data["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)

    assert "disclaimer" in data and len(data["disclaimer"]) > 0
    assert "advice" in data and len(data["advice"]) > 0
    assert any("xét nghiệm xác nhận" in str(a).lower() for a in data["advice"])
    assert "risk_index" in data
    assert "risk_level" in data


def test_predict_rejects_bmi_out_of_range_422(
    client: TestClient, sample_profile: dict
) -> None:
    """Kiểm tra /predict từ chối khi BMI nằm ngoài miền hợp lệ (BMI=5), thông điệp lỗi chứa 'BMI'."""
    payload = dict(sample_profile)
    payload["BMI"] = 5
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data
    errors = data["detail"]
    assert any("BMI" in err.get("message", "") and err.get("field") == "BMI" for err in errors)


def test_predict_rejects_extra_field_422(
    client: TestClient, sample_profile: dict
) -> None:
    """Kiểm tra /predict từ chối khi payload chứa trường lạ ngoài schema (extra forbidden)."""
    payload = dict(sample_profile)
    payload["unknown_extra_field"] = 999
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data


def test_predict_without_model_returns_503_friendly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_profile: dict
) -> None:
    """Kiểm tra khi tệp mô hình không tồn tại thì /predict trả về 503 với code 'MODEL_NOT_READY'."""
    missing_file = tmp_path / "non_existent_bundle.joblib"
    monkeypatch.setenv("MODEL_PATH", str(missing_file))
    reset_predictor()
    c = TestClient(app)
    res = c.post("/predict", json=sample_profile)
    assert res.status_code == 503
    data = res.json()
    assert data.get("detail", {}).get("code") == "MODEL_NOT_READY"


def test_glucose_260_forces_high_level_with_r1(
    client: TestClient, sample_profile: dict
) -> None:
    """Kiểm tra hồ sơ có glucose=260 bị ép mức nguy cơ lên 'cao' do kích hoạt luật an toàn R1."""
    payload = dict(sample_profile)
    payload.update(
        {
            "HighBP": 0,
            "HighChol": 0,
            "BMI": 19.0,
            "GenHlth": 1,
            "Age": 1,
            "Smoker": 0,
            "HeartDiseaseorAttack": 0,
            "Stroke": 0,
            "PhysActivity": 1,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "glucose": 260.0,
        }
    )
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] == "cao"
    alert_ids = [a["rule_id"] for a in data.get("alerts", [])]
    assert "R1" in alert_ids


def test_explain_has_note_and_whitelisted_what_if(
    client: TestClient, sample_profile: dict
) -> None:
    """Kiểm tra /explain trả về ghi chú chứa 'tương quan', kèm miễn trừ và kịch bản what-if trong danh sách trắng."""
    payload = {
        "profile": sample_profile,
        "what_if_changes": {"BMI": 22.0},
    }
    res = client.post("/explain", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "note" in data
    assert "tương quan" in data["note"]
    assert "disclaimer" in data
    assert "drivers" in data
    assert "what_if" in data
    assert len(data["what_if"]) > 0
    whitelisted = {"BMI", "PhysActivity", "Smoker", "HvyAlcoholConsump", "Fruits", "Veggies"}
    for scenario in data["what_if"]:
        assert scenario["feature"] in whitelisted
    assert data.get("custom_what_if") is not None


def test_cors_preflight_allows_local_origin(client: TestClient) -> None:
    """Kiểm tra yêu cầu OPTIONS preflight từ nguồn gốc http://localhost:5500 trả về tiêu đề CORS hợp lệ."""
    res = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5500"


def test_explain_rejects_disallowed_what_if_feature_422(client: TestClient, sample_profile: dict) -> None:
    """Khóa không nằm trong khối what_if (HighBP) phải bị từ chối bằng 422, không phải 500."""
    res = client.post("/explain", json={"profile": sample_profile, "what_if_changes": {"HighBP": 0}})
    assert res.status_code == 422
    assert "HighBP" in res.text


def test_explain_rejects_out_of_range_what_if_422(client: TestClient, sample_profile: dict) -> None:
    """BMI = 500 nằm ngoài miền → 422."""
    res = client.post("/explain", json={"profile": sample_profile, "what_if_changes": {"BMI": 500}})
    assert res.status_code == 422
    assert "BMI" in res.text


def test_explain_rejects_nan_what_if_422(client: TestClient, sample_profile: dict) -> None:
    """JSON chứa NaN phải bị từ chối bằng 422 và máy chủ không sập khi tuần tự hóa lỗi."""
    body = json.dumps({"profile": sample_profile, "what_if_changes": {"BMI": float("nan")}})
    res = client.post("/explain", content=body, headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_explain_accepts_every_key_the_ui_sends(client: TestClient, sample_profile: dict) -> None:
    """Sáu khóa lối sống hợp lệ (đúng bộ giao diện gửi) phải trả 200 và có custom_what_if."""
    changes = {"BMI": 24.5, "Smoker": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0}
    res = client.post("/explain", json={"profile": sample_profile, "what_if_changes": changes})
    assert res.status_code == 200
    assert res.json()["custom_what_if"] is not None

