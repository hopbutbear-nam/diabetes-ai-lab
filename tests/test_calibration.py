"""Kiểm thử các thuật toán hiệu chuẩn xác suất và lựa chọn phương pháp out-of-fold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diabetes_ai.evaluation.calibration import (
    IdentityCalibrator,
    IsotonicOvRCalibrator,
    LogitMultinomialCalibrator,
    fit_calibrator,
    load_calibrator,
    oof_calibrated,
    save_calibrator,
    select_calibration_method,
)
from diabetes_ai.evaluation.metrics import ece_ovr


def test_identity_calibrator_noop(tmp_path: Path) -> None:
    """Kiểm tra IdentityCalibrator giữ nguyên xác suất (chuẩn hóa tổng 1)."""
    rng = np.random.default_rng(42)
    proba = rng.dirichlet(alpha=[1, 1, 1], size=50)
    y = np.argmax(proba, axis=1)

    cal = IdentityCalibrator()
    cal.fit(proba, y)
    out = cal.predict(proba)

    assert out.shape == proba.shape
    assert np.allclose(out, proba)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert cal.method == "none"

    # Kiểm tra factory và lưu/nạp
    cal2 = fit_calibrator("none", proba, y)
    model_path = tmp_path / "identity.joblib"
    save_calibrator(cal2, model_path)
    loaded = load_calibrator(model_path)
    assert np.allclose(loaded.predict(proba), proba)


def test_logit_calibrator_reduces_ece() -> None:
    """Kiểm tra LogitMultinomialCalibrator giảm ECE khi xác suất ban đầu bị quá tự tin."""
    rng = np.random.default_rng(42)
    N = 1200
    y = rng.choice([0, 1, 2], size=N)
    noisy_guess = rng.choice([0, 1, 2], size=N)
    mask = rng.random(N) > 0.5
    noisy_guess[mask] = y[mask]

    # Giả lập xác suất quá tự tin (0.90 cho dự đoán nhưng độ chính xác chỉ khoảng 66%)
    proba = np.full((N, 3), 0.05)
    proba[np.arange(N), noisy_guess] = 0.90

    ece_before = ece_ovr(y, proba)["macro"]
    cal = LogitMultinomialCalibrator().fit(proba, y)
    cal_proba = cal.predict(proba)
    ece_after = ece_ovr(y, cal_proba)["macro"]

    assert ece_after < ece_before
    assert ece_after < 0.05
    assert np.allclose(cal_proba.sum(axis=1), 1.0)


def test_oof_shapes_and_sum_to_one() -> None:
    """Kiểm tra kích thước ma trận và ràng buộc tổng mỗi hàng bằng 1 với OOF."""
    rng = np.random.default_rng(42)
    N = 150
    y = rng.choice([0, 1, 2], size=N, p=[0.6, 0.3, 0.1])
    proba = rng.dirichlet(alpha=[2, 1, 1], size=N)

    for method in ("none", "logit_multinomial", "isotonic_ovr"):
        oof_p = oof_calibrated(method, proba, y, n_splits=3, seed=42)
        assert oof_p.shape == (N, 3)
        assert oof_p.dtype == np.float64
        assert np.allclose(oof_p.sum(axis=1), 1.0)
        assert np.all(oof_p >= 0.0)
        assert np.all(oof_p <= 1.0)


def test_select_method_picks_lowest_ece() -> None:
    """Kiểm tra hàm lựa chọn phương pháp chọn đúng ứng viên có ECE OOF nhỏ nhất."""
    rng = np.random.default_rng(42)
    N = 900
    y = rng.choice([0, 1, 2], size=N)
    noisy_guess = rng.choice([0, 1, 2], size=N)
    mask = rng.random(N) > 0.5
    noisy_guess[mask] = y[mask]

    proba = np.full((N, 3), 0.05)
    proba[np.arange(N), noisy_guess] = 0.90

    candidates = ("none", "logit_multinomial", "isotonic_ovr")
    res = select_calibration_method(
        proba,
        y,
        candidates=candidates,
        n_bins=10,
        n_splits=3,
        seed=42,
    )

    chosen = res["chosen"]
    assert chosen in candidates
    # Với dữ liệu bị quá tự tin, hiệu chuẩn phải tốt hơn không hiệu chuẩn
    assert chosen != "none"

    # Kiểm tra logic lựa chọn: candidate được chọn phải có ECE nhỏ nhất (hoặc Brier nhỏ hơn nếu bằng ECE)
    chosen_metrics = (res[chosen]["ece_macro_oof"], res[chosen]["brier_oof"])
    for cand in candidates:
        cand_metrics = (res[cand]["ece_macro_oof"], res[cand]["brier_oof"])
        assert chosen_metrics <= cand_metrics
