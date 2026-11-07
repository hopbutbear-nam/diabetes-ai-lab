"""Tìm kiếm siêu tham số cho mô hình, chỉ đánh giá trên tập validation.

Hỗ trợ Optuna nếu đã cài đặt; ngược lại tự động tìm kiếm ngẫu nhiên bằng NumPy có định seed.
Mục tiêu tối ưu hóa: AUC OvR macro trên tập validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from diabetes_ai.evaluation.metrics import roc_auc_ovr
from diabetes_ai.models.registry import build_model

try:
    import optuna  # type: ignore[import-untyped]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAVE_OPTUNA = True
except ImportError:
    HAVE_OPTUNA = False


def _sample_numpy(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Lấy mẫu ngẫu nhiên một bộ tham số từ không gian tìm kiếm space bằng NumPy."""
    params: dict[str, Any] = {}
    for param_name, spec in space.items():
        if isinstance(spec, list):
            # Danh sách các lựa chọn rời rạc
            idx = int(rng.integers(0, len(spec)))
            params[param_name] = spec[idx]
        elif isinstance(spec, tuple) and len(spec) == 2:
            low, high = spec
            if isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool):
                params[param_name] = int(rng.integers(low, high + 1))
            else:
                params[param_name] = float(rng.uniform(low, high))
        elif isinstance(spec, dict):
            spec_type = spec.get("type")
            if spec_type == "categorical" or "choices" in spec:
                choices = spec.get("choices", [])
                idx = int(rng.integers(0, len(choices)))
                params[param_name] = choices[idx]
            elif spec_type == "int" or (
                "low" in spec
                and "high" in spec
                and isinstance(spec["low"], int)
                and isinstance(spec["high"], int)
            ):
                low_val = int(spec["low"])
                high_val = int(spec["high"])
                params[param_name] = int(rng.integers(low_val, high_val + 1))
            elif spec_type == "float" or ("low" in spec and "high" in spec):
                low_val = float(spec["low"])
                high_val = float(spec["high"])
                if spec.get("log", False):
                    val = float(np.exp(rng.uniform(np.log(low_val), np.log(high_val))))
                else:
                    val = float(rng.uniform(low_val, high_val))
                params[param_name] = val
            else:
                params[param_name] = spec
        else:
            params[param_name] = spec
    return params


def tune(
    model_key: str,
    space: dict[str, Any],
    data: Any,
    n_trials: int = 20,
    seed: int = 42,
    force_random: bool = False,
    metrics_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Thực hiện tìm kiếm siêu tham số tối ưu AUC OvR macro trên tập validation.

    Args:
        model_key: Tên nhận diện mô hình ('majority', 'logreg', 'xgboost', ...).
        space: Không gian tìm kiếm siêu tham số.
        data: Đối tượng chứa dữ liệu train và validation (X_train, y_train, X_val, y_val).
        n_trials: Số lượng phép thử nghiệm (mặc định 20).
        seed: Seed sinh số ngẫu nhiên đảm bảo tính tái lập.
        force_random: Ép buộc dùng tìm kiếm ngẫu nhiên NumPy kể cả khi có Optuna.
        metrics_dir: Đường dẫn thư mục lưu báo cáo JSON kết quả tuning nếu có.

    Returns:
        dict[str, Any]: Gồm 'best_params', 'trials', và 'best_value'.
    """
    X_train = np.asarray(data.X_train)
    y_train = np.asarray(data.y_train)
    X_val = np.asarray(data.X_val)
    y_val = np.asarray(data.y_val)

    n_classes = len(np.unique(y_train))

    if HAVE_OPTUNA and not force_random:
        def objective(trial: Any) -> float:
            candidate_params: dict[str, Any] = {}
            for param_name, spec in space.items():
                if isinstance(spec, list):
                    candidate_params[param_name] = trial.suggest_categorical(param_name, spec)
                elif isinstance(spec, tuple) and len(spec) == 2:
                    low, high = spec
                    if isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool):
                        candidate_params[param_name] = trial.suggest_int(param_name, low, high)
                    else:
                        candidate_params[param_name] = trial.suggest_float(param_name, float(low), float(high))
                elif isinstance(spec, dict):
                    spec_type = spec.get("type")
                    if spec_type == "categorical" or "choices" in spec:
                        candidate_params[param_name] = trial.suggest_categorical(param_name, spec["choices"])
                    elif spec_type == "int":
                        candidate_params[param_name] = trial.suggest_int(param_name, int(spec["low"]), int(spec["high"]))
                    elif spec_type == "float":
                        candidate_params[param_name] = trial.suggest_float(
                            param_name,
                            float(spec["low"]),
                            float(spec["high"]),
                            log=bool(spec.get("log", False)),
                        )
                    else:
                        candidate_params[param_name] = spec
                else:
                    candidate_params[param_name] = spec

            model = build_model(
                model_key=model_key,
                params=candidate_params,
                n_classes=n_classes,
                seed=seed,
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)
            return float(roc_auc_ovr(y_val, proba))

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_score = float(study.best_value)
        trials = [
            {
                "trial_id": t.number,
                "params": t.params,
                "value": float(t.value) if t.value is not None else float("nan"),
                "score": float(t.value) if t.value is not None else float("nan"),
            }
            for t in study.trials
        ]
    else:
        rng = np.random.default_rng(seed)
        trials = []
        best_score = -float("inf")
        best_params = {}

        for trial_id in range(n_trials):
            candidate_params = _sample_numpy(space, rng)
            model = build_model(
                model_key=model_key,
                params=candidate_params,
                n_classes=n_classes,
                seed=seed + trial_id,
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)
            auc_val = float(roc_auc_ovr(y_val, proba))

            trial_rec = {
                "trial_id": trial_id,
                "params": candidate_params,
                "value": auc_val,
                "score": auc_val,
            }
            trials.append(trial_rec)

            if auc_val > best_score:
                best_score = auc_val
                best_params = dict(candidate_params)

    result = {
        "best_params": best_params,
        "trials": trials,
        "best_value": best_score,
    }

    if metrics_dir is not None:
        p = Path(metrics_dir)
        p.mkdir(parents=True, exist_ok=True)
        out_file = p / f"tuning_{model_key}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_key": model_key,
                    "best_params": best_params,
                    "best_value": best_score,
                    "trials": trials,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return result
