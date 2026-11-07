"""Kiểm thử kiểm định chéo K-fold (cross_validation) và tinh chỉnh siêu tham số (tuning)."""

from typing import Any

import numpy as np

from diabetes_ai.training.cross_validation import stratified_kfold_auc
from diabetes_ai.training.tuning import tune


class MockDataTrainOnly:
    """Đối tượng dữ liệu giả lập chỉ cho phép truy cập train; ném lỗi nếu đụng val/test."""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.X_train = X_train
        self.y_train = y_train

    @property
    def X_val(self) -> Any:
        raise AssertionError("Cross-validation không được truy cập X_val!")

    @property
    def y_val(self) -> Any:
        raise AssertionError("Cross-validation không được truy cập y_val!")

    @property
    def X_test(self) -> Any:
        raise AssertionError("Cross-validation không được truy cập X_test!")

    @property
    def y_test(self) -> Any:
        raise AssertionError("Cross-validation không được truy cập y_test!")


class MockDataTrainVal:
    """Đối tượng dữ liệu giả lập cho train và validation; cấm truy cập test."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    @property
    def X_test(self) -> Any:
        raise AssertionError("Tuning không được truy cập X_test!")

    @property
    def y_test(self) -> Any:
        raise AssertionError("Tuning không được truy cập y_test!")


def test_cv_uses_train_only() -> None:
    """Kiểm định chéo phân tầng chỉ chia trên train; tuyệt đối không đụng val/test."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((60, 4)).astype(np.float32)
    y_train = np.array([0, 1, 2] * 20, dtype=np.int64)

    data = MockDataTrainOnly(X_train=X_train, y_train=y_train)

    res = stratified_kfold_auc(
        model_key="logreg",
        params={"max_iter": 50},
        data=data,
        n_splits=3,
        seed=42,
    )

    assert "mean" in res
    assert "std" in res
    assert "folds" in res
    assert len(res["folds"]) == 3
    assert 0.0 <= res["mean"] <= 1.0
    assert res["std"] >= 0.0


def test_tune_random_fallback_deterministic() -> None:
    """Tìm kiếm ngẫu nhiên fallback bằng NumPy đảm bảo tính tái lập với cùng seed."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((60, 4)).astype(np.float32)
    y_train = np.array([0, 1, 2] * 20, dtype=np.int64)
    X_val = rng.standard_normal((30, 4)).astype(np.float32)
    y_val = np.array([0, 1, 2] * 10, dtype=np.int64)

    data = MockDataTrainVal(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    space = {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [50]}

    res1 = tune("logreg", space=space, data=data, n_trials=5, seed=42, force_random=True)
    res2 = tune("logreg", space=space, data=data, n_trials=5, seed=42, force_random=True)

    assert res1["best_params"] == res2["best_params"]
    assert res1["best_value"] == res2["best_value"]
    assert len(res1["trials"]) == len(res2["trials"])
    for t1, t2 in zip(res1["trials"], res2["trials"]):
        assert t1["params"] == t2["params"]
        assert t1["value"] == t2["value"]


def test_tune_returns_valid_params() -> None:
    """Tuning trả về tham số hợp lệ nằm trong search space và khớp với trial tốt nhất."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((60, 4)).astype(np.float32)
    y_train = np.array([0, 1, 2] * 20, dtype=np.int64)
    X_val = rng.standard_normal((30, 4)).astype(np.float32)
    y_val = np.array([0, 1, 2] * 10, dtype=np.int64)

    data = MockDataTrainVal(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    candidates = [0.01, 0.5, 2.0]
    space = {"C": candidates, "max_iter": [50]}

    res = tune("logreg", space=space, data=data, n_trials=4, seed=42, force_random=True)

    assert "best_params" in res
    assert "trials" in res
    assert res["best_params"]["C"] in candidates
    assert len(res["trials"]) == 4

    best_score_in_trials = max(t["value"] for t in res["trials"])
    assert res["best_value"] == best_score_in_trials
    matching_trials = [t for t in res["trials"] if t["value"] == res["best_value"]]
    assert any(t["params"] == res["best_params"] for t in matching_trials)
