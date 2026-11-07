"""Kiểm thử tính năng độ quan trọng hoán vị và SHAP (K07b)."""

import numpy as np
import pytest

from diabetes_ai.config import load_config
from diabetes_ai.data.schema import FEATURES
from diabetes_ai.explain.permutation import permutation_importance
from diabetes_ai.explain.shap_explain import explain_instance, shap_available
from diabetes_ai.explain.what_if import feature_effects


def test_permutation_ranks_informative_feature_first():
    """Kiểm tra hoán vị xếp đặc trưng duy nhất có thông tin lên đầu danh sách."""
    rng = np.random.default_rng(42)
    n = 200

    # 3 đặc trưng: f0 có tín hiệu mạnh, f1 và f2 chỉ là nhiễu ngẫu nhiên
    f0 = rng.normal(0, 1, size=n)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    X = np.column_stack([f0, f1, f2])

    # Nhãn y phụ thuộc hoàn toàn vào f0
    prob1 = 1.0 / (1.0 + np.exp(-3.0 * f0))
    y = (prob1 > 0.5).astype(int)

    def predict_fn(x_in: np.ndarray) -> np.ndarray:
        p1 = 1.0 / (1.0 + np.exp(-3.0 * x_in[:, 0]))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    features = ["informative_feature", "noise_1", "noise_2"]
    res = permutation_importance(predict_fn, X, y, features, n_repeats=5, seed=42)

    assert len(res) == 3
    assert res[0]["feature"] == "informative_feature"
    assert res[0]["mean_drop"] > 0.05
    assert res[0]["mean_drop"] > res[1]["mean_drop"]
    assert res[0]["mean_drop"] > res[2]["mean_drop"]


def test_permutation_deterministic():
    """Kiểm tra tính tái lập khi chạy hoán vị với cùng seed."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 2, size=60)

    def predict_fn(x_in: np.ndarray) -> np.ndarray:
        p1 = 1.0 / (1.0 + np.exp(-x_in[:, 0]))
        return np.column_stack([1.0 - p1, p1])

    features = ["f0", "f1", "f2", "f3"]
    res1 = permutation_importance(predict_fn, X, y, features, n_repeats=3, seed=42)
    res2 = permutation_importance(predict_fn, X, y, features, n_repeats=3, seed=42)

    assert res1 == res2


def test_feature_effects_skip_drop_list(sample_profile):
    """Kiểm tra feature_effects loại trừ hoàn toàn các đặc trưng trong drop_for_what_if."""
    cfg = load_config("features")
    drop_set = set(cfg.get("drop_for_what_if", []))
    assert len(drop_set) > 0, "Cần có danh sách drop_for_what_if trong configs/features.yaml."

    x_raw = np.array([sample_profile[f] for f in FEATURES], dtype=float)
    pop_mean_raw = np.zeros(len(FEATURES), dtype=float)

    def dummy_index_fn(X_mat: np.ndarray) -> np.ndarray:
        return np.sum(np.atleast_2d(X_mat), axis=1)

    effects = feature_effects(
        dummy_index_fn,
        x_raw,
        pop_mean_raw,
        features=list(FEATURES),
        skip=drop_set,
    )

    returned_features = [item["feature"] for item in effects]
    for dropped in drop_set:
        assert (
            dropped not in returned_features
        ), f"Đặc trưng {dropped} thuộc drop_for_what_if không được xuất hiện trong kết quả."

    assert len(returned_features) == len(FEATURES) - len(drop_set)


def test_shap_returns_values_or_none():
    """Kiểm tra explain_instance trả None nếu không có shap, hoặc dict đủ 21 khóa nếu có."""
    rng = np.random.default_rng(42)
    p = len(FEATURES)
    X_bg = rng.normal(size=(40, p))
    x = rng.normal(size=p)

    def predict_fn(x_in: np.ndarray) -> np.ndarray:
        mat = np.atleast_2d(x_in)
        s0 = mat[:, 0]
        s1 = mat[:, 1]
        s2 = mat[:, 2]
        scores = np.column_stack([s0, s1, s2])
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    if not shap_available():
        res = explain_instance(predict_fn, X_bg, x, list(FEATURES))
        assert res is None
    else:
        res = explain_instance(predict_fn, X_bg, x, list(FEATURES), class_index=-1)
        assert isinstance(res, dict)
        assert len(res) == len(FEATURES)
        for f in FEATURES:
            assert f in res
            assert isinstance(res[f], float)
