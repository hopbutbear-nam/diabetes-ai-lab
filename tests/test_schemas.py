"""Kiểm thử lược đồ Pydantic schemas.py và gói xuất mô hình export.py (K08b)."""

import json
from pathlib import Path
import numpy as np
import pytest
from pydantic import ValidationError

from diabetes_ai.data.schema import FEATURES
from diabetes_ai.inference.export import NumpyMLP, load_bundle, make_bundle, save_bundle
from diabetes_ai.inference.schemas import (
    Alert,
    ExplainRequest,
    ExplainResponse,
    PatientProfile,
    PredictionResponse,
)


@pytest.fixture
def sample_profile_dict():
    p = Path(__file__).parent / "fixtures" / "sample_profile.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def test_patient_profile_valid(sample_profile_dict):
    profile = PatientProfile(**sample_profile_dict)
    assert profile.BMI == 31.0
    assert profile.HighBP == 1
    assert profile.symptoms == []

    vec = profile.to_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (21,)
    assert vec[FEATURES.index("BMI")] == 31.0


def test_patient_profile_extra_field_forbidden(sample_profile_dict):
    data = dict(sample_profile_dict)
    data["extra_column"] = 123
    with pytest.raises(ValidationError):
        PatientProfile(**data)


def test_patient_profile_out_of_range(sample_profile_dict):
    data = dict(sample_profile_dict)
    data["BMI"] = 150.0  # Max is 100
    with pytest.raises(ValidationError):
        PatientProfile(**data)

    data2 = dict(sample_profile_dict)
    data2["HighBP"] = 2  # Binary must be 0 or 1
    with pytest.raises(ValidationError):
        PatientProfile(**data2)


def test_patient_profile_to_context(sample_profile_dict):
    data = dict(sample_profile_dict)
    data["glucose"] = 210.0
    data["fasting_glucose"] = 115.0
    data["symptoms"] = ["khát nước nhiều", "đi tiểu nhiều", "khát nước nhiều"]  # duplicate
    profile = PatientProfile(**data)

    ctx = profile.to_context(p_diabetes=0.6, risk_index=0.7)
    assert ctx["glucose"] == 210.0
    assert ctx["fasting_glucose"] == 115.0
    assert ctx["n_symptoms"] == 2  # deduplicated
    assert ctx["high_bp"] == 1
    assert ctx["bmi"] == 31.0
    assert ctx["p_diabetes"] == 0.6
    assert ctx["risk_index"] == 0.7


def test_numpy_mlp_softmax_and_relu():
    # 2 lớp: in=21 -> hidden=10 -> out=3
    np.random.seed(42)
    w1 = np.random.randn(10, 21).astype(np.float32)
    b1 = np.zeros(10, dtype=np.float32)
    w2 = np.random.randn(3, 10).astype(np.float32)
    b2 = np.zeros(3, dtype=np.float32)

    mlp = NumpyMLP(weights=[w1, w2], biases=[b1, b2])
    x = np.random.randn(4, 21).astype(np.float32)
    probs = mlp.predict_proba(x)

    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)
    preds = mlp.predict(x)
    assert len(preds) == 4
    np.testing.assert_array_equal(preds, np.argmax(probs, axis=1))


def test_bundle_save_load(tmp_path):
    bundle = make_bundle(
        model=None,
        scaler={"mean": [0.0] * 21, "std": [1.0] * 21},
        class_names=["không", "tiểu đường"],
        task="binary",
        variant="full",
        run_name="test_run",
    )
    assert bundle["schema_version"] == 1
    assert len(bundle["feature_order"]) == 21
    assert "disclaimer" in bundle

    out_file = tmp_path / "test_bundle.joblib"
    save_bundle(bundle, out_file)
    assert out_file.is_file()

    loaded = load_bundle(out_file)
    assert loaded["run_name"] == "test_run"
    assert loaded["task"] == "binary"
    assert loaded["feature_order"] == FEATURES
