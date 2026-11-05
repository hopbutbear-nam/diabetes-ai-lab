"""Kiểm tra tổng xác suất dự đoán của các mô hình luôn bằng 1 ± 1e-6."""

import numpy as np
import pytest

from diabetes_ai.data.split import make_split
from diabetes_ai.features.build_features import build_experiment_data_from_frame
from diabetes_ai.models.baselines import LogisticRegressionModel, MajorityClassifier
from diabetes_ai.models.xgboost_model import XGBoostModel


def test_majority_proba_sums_to_one(synthetic_df) -> None:
    """Kiểm tra MajorityClassifier: tổng xác suất = 1, shape (n, 3) và giá trị trong [0, 1]."""
    split = make_split(synthetic_df, seed=42)
    exp_data = build_experiment_data_from_frame(synthetic_df, split, task="multiclass")

    model = MajorityClassifier(n_classes=3, seed=42)
    model.fit(exp_data.X_train, exp_data.y_train, X_val=exp_data.X_val, y_val=exp_data.y_val)

    p = model.predict_proba(exp_data.X_val)

    assert p.shape == (len(exp_data.X_val), 3)
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-6
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)

    preds = model.predict(exp_data.X_val)
    assert np.array_equal(preds, np.argmax(p, axis=1))


def test_logreg_proba_sums_to_one(synthetic_df) -> None:
    """Kiểm tra LogisticRegression: tổng xác suất = 1, shape (n, 3) và giá trị trong [0, 1]."""
    split = make_split(synthetic_df, seed=42)
    exp_data = build_experiment_data_from_frame(synthetic_df, split, task="multiclass")

    model = LogisticRegressionModel(params={"max_iter": 50}, n_classes=3, seed=42)
    model.fit(exp_data.X_train, exp_data.y_train, X_val=exp_data.X_val, y_val=exp_data.y_val)

    p = model.predict_proba(exp_data.X_val)

    assert p.shape == (len(exp_data.X_val), 3)
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-6
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)


def test_xgb_or_hgb_proba_sums_to_one(synthetic_df) -> None:
    """Kiểm tra XGBoost (hoặc HGB dự phòng): tổng xác suất = 1, shape (n, 3) và giá trị trong [0, 1]."""
    split = make_split(synthetic_df, seed=42)
    exp_data = build_experiment_data_from_frame(synthetic_df, split, task="multiclass")

    model = XGBoostModel(params={"n_estimators": 10, "max_depth": 3}, n_classes=3, seed=42)
    model.fit(exp_data.X_train, exp_data.y_train, X_val=exp_data.X_val, y_val=exp_data.y_val)

    p = model.predict_proba(exp_data.X_val)

    assert p.shape == (len(exp_data.X_val), 3)
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-6
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)
    assert "backend" in model.fit_info


def test_binary_logreg_has_two_columns_and_sums_to_one(synthetic_df) -> None:
    """Kiểm tra mô hình phân loại nhị phân: shape (n, 2) và tổng xác suất = 1."""
    split = make_split(synthetic_df, seed=42)
    exp_data = build_experiment_data_from_frame(synthetic_df, split, task="binary")

    model = LogisticRegressionModel(params={"max_iter": 50}, n_classes=2, seed=42)
    model.fit(exp_data.X_train, exp_data.y_train, X_val=exp_data.X_val, y_val=exp_data.y_val)

    p = model.predict_proba(exp_data.X_val)

    assert p.shape == (len(exp_data.X_val), 2)
    assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-6
    assert np.all(p >= 0.0)
    assert np.all(p <= 1.0)
