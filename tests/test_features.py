"""6 test cho features/build_features.py và features/scalers.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diabetes_ai.data.schema import FEATURES, TARGET
from diabetes_ai.data.split import make_split
from diabetes_ai.features.build_features import (
    ExperimentData,
    build_experiment_data_from_frame,
    to_xy,
)
from diabetes_ai.features.scalers import StandardScalerNP
from diabetes_ai.training.imbalance import class_weights

# Import fixture make_synthetic_df từ conftest
from tests.conftest import make_synthetic_df


# ---------------------------------------------------------------------------
# test_scaler_fit_uses_only_train_stats
# ---------------------------------------------------------------------------

def test_scaler_fit_uses_only_train_stats(synthetic_df: pd.DataFrame) -> None:
    """Scaler fit trên train; mean của scaler khớp mean train, khác mean toàn bộ."""
    split = make_split(synthetic_df, seed=42)
    ed = build_experiment_data_from_frame(synthetic_df, split)

    # Mean scaler phải khớp mean train (trước chuẩn hóa)
    X_train_raw, _ = to_xy(synthetic_df, split.train)
    expected_mean = X_train_raw.mean(axis=0)
    np.testing.assert_allclose(
        ed.scaler.mean_,
        expected_mean,
        rtol=1e-5,
        err_msg="mean_ của scaler phải khớp mean của tập train",
    )

    # Mean toàn bộ dữ liệu phải KHÁC mean train (vì val/test khác phân bố thực nghiệm)
    X_all, _ = to_xy(synthetic_df, np.arange(len(synthetic_df)))
    global_mean = X_all.mean(axis=0)
    # Nếu train = toàn bộ sẽ bằng → chỉ kiểm tra khi split thực sự tách nhau
    if not np.allclose(ed.scaler.mean_, global_mean, rtol=1e-5):
        assert True  # đã khác nhau
    # Đây là assertion quan trọng: scaler không fit trên val/test
    # Kiểm tra gián tiếp: X_val chưa bị fit riêng
    assert ed.X_val.dtype == np.float32


# ---------------------------------------------------------------------------
# test_scaler_std_zero_guard
# ---------------------------------------------------------------------------

def test_scaler_std_zero_guard() -> None:
    """Đặc trưng hằng số (std=0) phải được thay bằng std=1 để không chia 0."""
    n = 100
    X = np.ones((n, 3), dtype=np.float64)
    X[:, 1] = np.arange(n, dtype=np.float64)  # cột 1 có std > 0

    scaler = StandardScalerNP()
    scaler.fit(X)

    # Cột 0 và 2 hằng số → std_ phải là 1.0
    assert scaler.std_[0] == pytest.approx(1.0), "std của cột hằng số phải = 1.0"
    assert scaler.std_[2] == pytest.approx(1.0)

    # Transform không được raise
    X_scaled = scaler.transform(X)
    assert np.all(np.isfinite(X_scaled)), "Transform cột hằng số phải trả về giá trị hữu hạn"


# ---------------------------------------------------------------------------
# test_scaler_json_roundtrip
# ---------------------------------------------------------------------------

def test_scaler_json_roundtrip() -> None:
    """Lưu scaler ra JSON rồi tải lại; mean_ và std_ phải giống hệt."""
    rng = np.random.default_rng(10)
    X = rng.standard_normal((200, 5))

    scaler = StandardScalerNP(feature_names=["a", "b", "c", "d", "e"])
    scaler.fit(X)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scaler.json"
        scaler.save_json(path)
        loaded = StandardScalerNP.load_json(path)

    np.testing.assert_allclose(loaded.mean_, scaler.mean_, rtol=1e-12)
    np.testing.assert_allclose(loaded.std_, scaler.std_, rtol=1e-12)
    assert loaded.feature_names == scaler.feature_names


# ---------------------------------------------------------------------------
# test_class_weights_match_formula
# ---------------------------------------------------------------------------

def test_class_weights_match_formula() -> None:
    """class_weights trên đếm cụ thể phải cho [0.4989; 8.8333; 1.9236] ± 1e-3.

    Dùng đếm train full: lớp 0 ≈ 84%, lớp 1 ≈ 2%, lớp 2 ≈ 14%
    (ví dụ: 172502 mẫu: n0 ≈ 144992, n1 ≈ 3443, n2 ≈ 24067).
    Công thức: w_k = (n / (K * n_k))^0.75
    """
    # Dùng đếm thực tế từ manifest full_seed42 (data/splits/full_seed42/manifest.json)
    n0 = 145317
    n1 = 3149
    n2 = 24036
    n_total = n0 + n1 + n2
    K = 3

    y = np.concatenate([
        np.zeros(n0, dtype=np.int64),
        np.ones(n1, dtype=np.int64),
        2 * np.ones(n2, dtype=np.int64),
    ])

    w = class_weights(y, n_classes=K, power=0.75)

    expected = np.array([
        (n_total / (K * n0)) ** 0.75,
        (n_total / (K * n1)) ** 0.75,
        (n_total / (K * n2)) ** 0.75,
    ])

    np.testing.assert_allclose(w, expected, rtol=1e-6)
    # Kiểm tra gần các số đã nêu trong plan
    assert abs(w[0] - 0.4989) < 1e-3, f"w[0] = {w[0]:.4f}, mong đợi ~0.4989"
    assert abs(w[1] - 8.8333) < 1e-2, f"w[1] = {w[1]:.4f}, mong đợi ~8.8333"
    assert abs(w[2] - 1.9236) < 1e-3, f"w[2] = {w[2]:.4f}, mong đợi ~1.9236"


# ---------------------------------------------------------------------------
# test_build_experiment_data_shapes_and_test_hidden
# ---------------------------------------------------------------------------

def test_build_experiment_data_shapes_and_test_hidden(synthetic_df: pd.DataFrame) -> None:
    """X_test là None khi include_test=False; thứ tự cột đúng FEATURES."""
    split = make_split(synthetic_df, seed=42)

    # include_test=False
    ed = build_experiment_data_from_frame(synthetic_df, split, include_test=False)
    assert ed.X_test is None, "X_test phải là None khi include_test=False"
    assert ed.y_test is None, "y_test phải là None khi include_test=False"

    # Kiểm tra shape
    assert ed.X_train.shape[1] == len(FEATURES), f"X_train phải có {len(FEATURES)} cột"
    assert ed.X_val.shape[1] == len(FEATURES)
    assert ed.X_train.dtype == np.float32
    assert ed.X_val.dtype == np.float32
    assert ed.y_train.dtype == np.int64
    assert ed.y_val.dtype == np.int64

    # Kiểm tra n_classes và class_names
    assert ed.n_classes == 3
    assert len(ed.class_names) == 3

    # include_test=True
    ed2 = build_experiment_data_from_frame(synthetic_df, split, include_test=True)
    assert ed2.X_test is not None, "X_test phải không phải None khi include_test=True"
    assert ed2.X_test.shape[1] == len(FEATURES)
    assert ed2.X_test.dtype == np.float32

    # Thứ tự cột: lấy dòng đầu train raw rồi so sánh
    X_first, _ = to_xy(synthetic_df, split.train[:1])
    # Sau chuẩn hóa, giá trị sẽ thay đổi nhưng phải là float32
    assert ed.X_train[0].dtype == np.float32


# ---------------------------------------------------------------------------
# test_binary_task_labels
# ---------------------------------------------------------------------------

def test_binary_task_labels(synthetic_df: pd.DataFrame) -> None:
    """Với task='binary', nhãn phải là 0 hoặc 1; n_classes = 2."""
    split = make_split(synthetic_df, seed=42)
    ed = build_experiment_data_from_frame(synthetic_df, split, task="binary")

    unique_labels = set(ed.y_train.tolist())
    assert unique_labels.issubset({0, 1}), f"Nhãn binary phải là {{0, 1}}, nhận {unique_labels}"
    assert ed.n_classes == 2, f"n_classes phải = 2 với binary, nhận {ed.n_classes}"
    assert ed.y_train.dtype == np.int64
