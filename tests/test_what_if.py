"""Kiểm thử tính năng tác động yếu tố và what-if lối sống (K07a)."""

from typing import Any
import numpy as np
import pytest

from diabetes_ai.config import load_config
from diabetes_ai.data.schema import FEATURES
from diabetes_ai.explain.what_if import (
    WHAT_IF_NOTE,
    feature_effects,
    what_if_custom,
    what_if_scenarios,
)


@pytest.fixture
def linear_index_fn():
    """Hàm tính chỉ số nguy cơ tuyến tính giả lập nhận ma trận thô chưa chuẩn hóa."""
    weights = np.zeros(len(FEATURES), dtype=float)
    feat_map = {f: i for i, f in enumerate(FEATURES)}

    # Trọng số dương: tăng giá trị -> tăng nguy cơ
    weights[feat_map["BMI"]] = 0.04
    weights[feat_map["Smoker"]] = 0.25
    weights[feat_map["HvyAlcoholConsump"]] = 0.15
    weights[feat_map["HighBP"]] = 0.30
    weights[feat_map["HighChol"]] = 0.20
    weights[feat_map["Age"]] = 0.05
    weights[feat_map["GenHlth"]] = 0.10

    # Trọng số âm: có hành vi bảo vệ -> giảm nguy cơ
    weights[feat_map["PhysActivity"]] = -0.15
    weights[feat_map["Fruits"]] = -0.08
    weights[feat_map["Veggies"]] = -0.06

    def _fn(X_raw: np.ndarray) -> np.ndarray:
        mat = np.atleast_2d(X_raw)
        return mat @ weights

    return _fn


def test_what_if_sorted_desc_by_delta(linear_index_fn, sample_profile):
    """Kiểm tra các kịch bản what-if được sắp xếp giảm dần theo delta_index."""
    cfg = load_config("features")
    profile = dict(sample_profile)
    # Đặt các yếu tố lối sống ở mức chưa lành mạnh
    profile["BMI"] = 34.0
    profile["Smoker"] = 1
    profile["PhysActivity"] = 0
    profile["Fruits"] = 0
    profile["Veggies"] = 0
    profile["HvyAlcoholConsump"] = 1

    x_raw = np.array([profile[f] for f in FEATURES], dtype=float)
    scenarios = what_if_scenarios(linear_index_fn, x_raw, cfg)

    assert len(scenarios) > 0, "Phải có ít nhất một kịch bản cải thiện lối sống."

    deltas = [s["delta_index"] for s in scenarios]
    assert deltas == sorted(deltas, reverse=True), "Kịch bản phải sắp giảm dần theo delta_index."

    for item in scenarios:
        assert "feature" in item
        assert "action" in item
        assert "from_value" in item
        assert "to_value" in item
        assert "delta_index" in item
        assert "note" in item
        assert item["note"] == WHAT_IF_NOTE
        assert "tương quan" in item["note"]


def test_what_if_skips_already_healthy(linear_index_fn, sample_profile):
    """Kiểm tra bỏ qua đặc trưng đã đạt ngưỡng lành mạnh (BMI 24 không có kịch bản BMI)."""
    cfg = load_config("features")
    profile = dict(sample_profile)

    # BMI 24.0: target là 24.5, min_gap là 0.5 -> 24.0 <= 24.5 + 0.5 -> bỏ qua BMI
    profile["BMI"] = 24.0
    profile["Smoker"] = 1  # Chưa lành mạnh
    profile["PhysActivity"] = 1  # Đã lành mạnh
    profile["Fruits"] = 1  # Đã lành mạnh
    profile["Veggies"] = 1  # Đã lành mạnh
    profile["HvyAlcoholConsump"] = 0  # Đã lành mạnh

    x_raw = np.array([profile[f] for f in FEATURES], dtype=float)
    scenarios = what_if_scenarios(linear_index_fn, x_raw, cfg)

    features_in_scenarios = [s["feature"] for s in scenarios]
    assert "BMI" not in features_in_scenarios, "BMI 24 không được xuất hiện trong kịch bản BMI."
    assert "PhysActivity" not in features_in_scenarios
    assert "Fruits" not in features_in_scenarios
    assert "Veggies" not in features_in_scenarios
    assert "HvyAlcoholConsump" not in features_in_scenarios
    assert "Smoker" in features_in_scenarios

    # Khi toàn bộ đã lành mạnh
    profile["Smoker"] = 0
    x_all_healthy = np.array([profile[f] for f in FEATURES], dtype=float)
    healthy_scenarios = what_if_scenarios(linear_index_fn, x_all_healthy, cfg)
    assert len(healthy_scenarios) == 0, "Không được sinh kịch bản nào khi đã lành mạnh toàn bộ."


def test_what_if_respects_allowed_features_and_drop_list(linear_index_fn, sample_profile):
    """Kiểm tra kịch bản không chứa Education, Income... và what_if_custom từ chối Age."""
    cfg = load_config("features")
    what_if_cfg = cfg.get("what_if", {})
    allowed = set(what_if_cfg.keys())
    drop_list = set(cfg.get("drop_for_what_if", []))

    profile = dict(sample_profile)
    profile["BMI"] = 32.0
    profile["Smoker"] = 1
    profile["Education"] = 1
    profile["Income"] = 1
    x_raw = np.array([profile[f] for f in FEATURES], dtype=float)

    # 1. what_if_scenarios không có Education, Income...
    scenarios = what_if_scenarios(linear_index_fn, x_raw, cfg)
    scenario_features = {s["feature"] for s in scenarios}
    assert "Education" not in scenario_features
    assert "Income" not in scenario_features
    assert "Age" not in scenario_features
    assert not scenario_features.intersection(drop_list)
    assert scenario_features.issubset(allowed)

    # 2. feature_effects loại trừ các đặc trưng trong drop_for_what_if
    pop_mean = np.zeros(len(FEATURES), dtype=float)
    effects = feature_effects(linear_index_fn, x_raw, pop_mean, skip=drop_list)
    effect_features = {e["feature"] for e in effects}
    assert not effect_features.intersection(drop_list)
    assert "Education" not in effect_features
    assert "Income" not in effect_features

    # 3. what_if_custom từ chối đặc trưng ngoài allowed (ví dụ Age)
    with pytest.raises(ValueError, match="Chỉ được thay đổi:"):
        what_if_custom(linear_index_fn, x_raw, {"Age": 5.0}, allowed=allowed)


def test_what_if_custom_changes(linear_index_fn, sample_profile):
    """Kiểm tra tính toán what_if_custom và kiểm tra miền giá trị hợp lệ."""
    cfg = load_config("features")
    allowed = set(cfg["what_if"].keys())

    profile = dict(sample_profile)
    profile["BMI"] = 33.0
    profile["Smoker"] = 1
    x_raw = np.array([profile[f] for f in FEATURES], dtype=float)

    # Thay đổi hợp lệ: giảm BMI về 24.5 và bỏ thuốc lá
    changes = {"BMI": 24.5, "Smoker": 0.0}
    res = what_if_custom(linear_index_fn, x_raw, changes, allowed=allowed)

    assert "before" in res
    assert "after" in res
    assert "delta" in res
    assert "note" in res
    assert res["note"] == WHAT_IF_NOTE
    assert "tương quan" in res["note"]
    assert res["before"] > res["after"], "Giảm BMI và bỏ thuốc lá phải làm giảm chỉ số nguy cơ."
    assert np.isclose(res["delta"], res["before"] - res["after"])

    # Giá trị ngoài FEATURE_RANGES phải báo lỗi ValueError
    with pytest.raises(ValueError, match="nằm ngoài khoảng cho phép"):
        what_if_custom(linear_index_fn, x_raw, {"BMI": 150.0}, allowed=allowed)

    with pytest.raises(ValueError, match="nằm ngoài khoảng cho phép"):
        what_if_custom(linear_index_fn, x_raw, {"Smoker": 2.0}, allowed=allowed)
