"""3 test cho pipeline huấn luyện (K03b): ghi artifact, tính tái lập và không nạp test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from diabetes_ai.data.split import make_split
from diabetes_ai.features.build_features import build_experiment_data_from_frame
from diabetes_ai.models.registry import load_run
from diabetes_ai.training.train import run_training
from tests.conftest import make_synthetic_df


@pytest.fixture
def synthetic_experiment_data():
    """Tạo ExperimentData giả lập không nạp tập test."""
    df = make_synthetic_df(n=600, seed=42)
    split = make_split(df, seed=42)
    return build_experiment_data_from_frame(df, split, include_test=False)


def test_run_training_writes_artifacts(
    tmp_path: Path, synthetic_experiment_data
) -> None:
    """Kiểm tra pipeline tạo đủ artifact: model.joblib, scaler.json, meta.json, train_<run>.json."""
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"

    model_cfg = {
        "model": "logreg",
        "name": "logreg_pipe_test",
        "params": {"max_iter": 50, "C": 1.0},
    }

    res = run_training(
        model_config=model_cfg,
        data=synthetic_experiment_data,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        log_to_experiment_log=False,
    )

    run_name = res["run_name"]
    run_dir = models_dir / run_name

    assert (run_dir / "model.joblib").is_file(), "Thiếu file model.joblib"
    assert (run_dir / "scaler.json").is_file(), "Thiếu file scaler.json"
    assert (run_dir / "meta.json").is_file(), "Thiếu file meta.json"
    assert (metrics_dir / f"train_{run_name}.json").is_file(), f"Thiếu train_{run_name}.json"

    # Kiểm tra load_run hoạt động đúng với artifact vừa ghi
    loaded = load_run(run_name, models_dir=models_dir)
    assert loaded.model is not None
    assert loaded.scaler is not None
    assert loaded.meta["run_name"] == run_name

    p = loaded.predict_proba_scaled(synthetic_experiment_data.X_val)
    assert p.shape == (len(synthetic_experiment_data.X_val), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)

    labels = loaded.predict_labels(p)
    assert len(labels) == len(synthetic_experiment_data.X_val)


def test_run_training_reproducible(
    tmp_path: Path, synthetic_experiment_data
) -> None:
    """Hai lần huấn luyện cùng seed phải cho kết quả AUC val chênh lệch <= 1e-6."""
    model_cfg = {
        "model": "logreg",
        "name": "logreg_repro_test",
        "params": {"max_iter": 50, "C": 1.0},
    }

    res1 = run_training(
        model_config=model_cfg,
        seed=42,
        data=synthetic_experiment_data,
        models_dir=tmp_path / "models_1",
        metrics_dir=tmp_path / "metrics_1",
        log_to_experiment_log=False,
    )

    res2 = run_training(
        model_config=model_cfg,
        seed=42,
        data=synthetic_experiment_data,
        models_dir=tmp_path / "models_2",
        metrics_dir=tmp_path / "metrics_2",
        log_to_experiment_log=False,
    )

    auc1 = res1["val"]["roc_auc_ovr_macro"]
    auc2 = res2["val"]["roc_auc_ovr_macro"]

    assert abs(auc1 - auc2) <= 1e-6, f"AUC không tái lập: {auc1} vs {auc2}"


def test_run_training_never_loads_test(
    tmp_path: Path, synthetic_experiment_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Đảm bảo pipeline không bao giờ nạp test (data.X_test is None, include_test=False)."""
    assert synthetic_experiment_data.X_test is None
    assert synthetic_experiment_data.y_test is None

    called_kwargs: dict[str, Any] = {}

    def mock_load_experiment_data(**kwargs: Any):
        called_kwargs.update(kwargs)
        return synthetic_experiment_data

    monkeypatch.setattr(
        "diabetes_ai.training.train.load_experiment_data", mock_load_experiment_data
    )

    model_cfg = {
        "model": "logreg",
        "name": "logreg_no_test",
        "params": {"max_iter": 50, "C": 1.0},
    }

    run_training(
        model_config=model_cfg,
        data=None,  # Ép gọi load_experiment_data
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        log_to_experiment_log=False,
    )

    assert (
        called_kwargs.get("include_test") is False
    ), "load_experiment_data phải được gọi với include_test=False"
