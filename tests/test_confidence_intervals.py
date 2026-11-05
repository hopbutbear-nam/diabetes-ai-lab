"""Kiểm thử khoảng tin cậy Wilson và Bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from diabetes_ai.evaluation.confidence_intervals import (
    bootstrap_ci,
    bootstrap_multi,
    stratified_bootstrap_indices,
    wilson_interval,
)
from diabetes_ai.evaluation.metrics import roc_auc_ovr


def test_wilson_known_values() -> None:
    """Kiểm tra các giá trị Wilson đã biết theo đặc tả."""
    # wilson(50, 100) ≈ (0.4038, 0.5962) ± 1e-3
    low_50, high_50 = wilson_interval(50, 100)
    assert abs(low_50 - 0.4038) <= 1e-3
    assert abs(high_50 - 0.5962) <= 1e-3

    # wilson(0, 10) = (0.0, 0.2775) ± 1e-3
    low_0, high_0 = wilson_interval(0, 10)
    assert abs(low_0 - 0.0) <= 1e-3
    assert abs(high_0 - 0.2775) <= 1e-3

    # Biên n = 0
    low_empty, high_empty = wilson_interval(0, 0)
    assert low_empty == 0.0
    assert high_empty == 1.0


def test_bootstrap_ci_contains_point() -> None:
    """Kiểm tra khoảng tin cậy bootstrap bao chứa điểm ước lượng point."""
    rng = np.random.default_rng(42)
    n_samples = 300
    y_true = rng.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    raw_scores = rng.normal(size=(n_samples, 3))
    exp_scores = np.exp(raw_scores)
    proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    y_pred = np.argmax(proba, axis=1)

    # bootstrap_ci
    ci_res = bootstrap_ci(
        y_true,
        proba,
        metric_fn=roc_auc_ovr,
        n=200,
        ci=0.95,
        seed=42,
        stratified=True,
    )
    assert ci_res["low"] <= ci_res["point"] <= ci_res["high"]
    assert ci_res["n_boot"] == 200

    # bootstrap_multi
    multi_res = bootstrap_multi(
        y_true,
        proba,
        y_pred,
        n=200,
        ci=0.95,
        seed=42,
    )
    auc_res = multi_res["roc_auc_ovr_macro"]
    assert auc_res["low"] <= auc_res["point"] <= auc_res["high"]

    rec_res = multi_res["recall_per_class"]
    for c in range(3):
        assert rec_res["low"][c] <= rec_res["point"][c] <= rec_res["high"][c]


def test_bootstrap_deterministic() -> None:
    """Kiểm tra tính tái lập khi cùng seed."""
    rng = np.random.default_rng(99)
    y_true = rng.choice([0, 1, 2], size=150, p=[0.6, 0.3, 0.1])
    proba = rng.dirichlet(alpha=[1, 1, 1], size=150)
    y_pred = np.argmax(proba, axis=1)

    res1 = bootstrap_ci(y_true, proba, metric_fn=roc_auc_ovr, n=100, seed=42)
    res2 = bootstrap_ci(y_true, proba, metric_fn=roc_auc_ovr, n=100, seed=42)
    assert res1 == res2

    multi1 = bootstrap_multi(y_true, proba, y_pred, n=100, seed=42)
    multi2 = bootstrap_multi(y_true, proba, y_pred, n=100, seed=42)
    assert multi1 == multi2


def test_bootstrap_stratified_counts() -> None:
    """Kiểm tra lấy mẫu phân tầng giữ đúng số lượng mẫu mỗi lớp."""
    y = np.array([0] * 70 + [1] * 20 + [2] * 10)
    rng = np.random.default_rng(123)

    idx = stratified_bootstrap_indices(y, rng)
    assert len(idx) == len(y)
    assert np.array_equal(np.bincount(y[idx]), np.bincount(y))
