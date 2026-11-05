"""Pipeline huấn luyện mô hình: nạp cấu hình, huấn luyện, đánh giá val, lưu artifact.

Tuân thủ nghiêm ngặt nguyên tắc: chỉ nạp train và validation, tuyệt đối không nạp test.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import sklearn

from ..config import load_config, load_yaml
from ..evaluation.metrics import compute_all
from ..features.build_features import ExperimentData, load_experiment_data
from ..logging_utils import get_logger
from ..models.registry import build_model
from ..paths import CONFIGS_DIR, DOCS_DIR, METRICS_DIR, MODELS_DIR
from ..seed import set_seed
from .imbalance import class_weights, sample_weights

logger = get_logger(__name__)


def append_experiment_log(
    date: str,
    model: str,
    config: str,
    auc: float,
    macro_f1: float,
    note: str,
    path: Path | None = None,
) -> None:
    """Thêm một dòng nhật ký thí nghiệm vào file docs/experiment_log.md.

    Cột AUC và Macro-F1 được định dạng 0.xxx (val). Dấu '|' trong ghi chú được thay bằng '/'.

    Args:
        date: Ngày thí nghiệm (YYYY-MM-DD).
        model: Tên mô hình.
        config: Tên hoặc đường dẫn file config.
        auc: AUC OvR macro trên tập validation.
        macro_f1: Macro-F1 trên tập validation.
        note: Ghi chú giải thích.
        path: Đường dẫn file nhật ký (mặc định docs/experiment_log.md).
    """
    log_path = Path(path) if path is not None else DOCS_DIR / "experiment_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    clean_note = str(note).replace("|", "/").strip()
    auc_str = f"{auc:.3f} (val)" if auc is not None and not np.isnan(auc) else "- (val)"
    f1_str = (
        f"{macro_f1:.3f} (val)"
        if macro_f1 is not None and not np.isnan(macro_f1)
        else "- (val)"
    )
    row = f"| {date} | {model} | {config} | {auc_str} | {f1_str} | {clean_note} |\n"

    header = (
        "# Nhật ký thí nghiệm\n\n"
        "| Ngày | Mô hình | Config | AUC | Macro-F1 | Ghi chú |\n"
        "|---|---|---|---|---|---|\n"
    )

    if not log_path.is_file():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header + row)
        return

    with open(log_path, "r", encoding="utf-8") as f:
        existing = f.read()

    # Chuẩn hoá nếu header có ký tự phân cách escape dạng chuỗi
    if r"|\n|---|---|---|---|---|---|" in existing:
        existing = existing.replace(
            r"|\n|---|---|---|---|---|---|",
            "|\n|---|---|---|---|---|---|",
        )

    if not existing.endswith("\n"):
        existing += "\n"

    new_content = existing + row
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def run_training(
    model_config: dict[str, Any] | Path | str,
    task: str | None = None,
    dedup: bool | None = None,
    seed: int | None = None,
    max_train_rows: int | None = None,
    run_suffix: str = "",
    data: ExperimentData | None = None,
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
    log_to_experiment_log: bool = True,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline huấn luyện cho một mô hình.

    Các bước:
    1. Nạp cấu hình (task, dedup mặc định từ configs/data.yaml).
    2. Thiết lập hạt giống (set_seed).
    3. Nạp dữ liệu qua load_experiment_data(..., include_test=False).
    4. Tính trọng số mẫu nếu imbalance.strategy == 'class_weight'.
    5. Khởi tạo mô hình và huấn luyện (fit), đo thời gian huấn luyện.
    6. Dự đoán xác suất val, tính bộ chỉ số (compute_all).
    7. Lưu mô hình, scaler.json, meta.json vào models/<run>/.
    8. Ghi kết quả vào reports/metrics/train_<run>.json.
    9. Nếu bật, ghi nhật ký vào docs/experiment_log.md.

    Args:
        model_config: Dict cấu hình hoặc đường dẫn file YAML.
        task: 'multiclass' hoặc 'binary'. Nếu None đọc từ data.yaml.
        dedup: Có dùng biến thể khử trùng không. Nếu None đọc từ data.yaml.
        seed: Seed sinh số ngẫu nhiên.
        max_train_rows: Giới hạn số dòng tập huấn luyện (dùng để test nhanh).
        run_suffix: Hậu tố gắn thêm vào tên run.
        data: ExperimentData đã nạp sẵn (nếu có).
        models_dir: Thư mục lưu artifacts mô hình.
        metrics_dir: Thư mục lưu kết quả metrics JSON.
        log_to_experiment_log: Có ghi vào docs/experiment_log.md hay không.

    Returns:
        dict: Chứa {run_name, meta, val, fit_info}.
    """
    # 1. Nạp cấu hình
    if isinstance(model_config, (str, Path)):
        config_path_str = str(model_config)
        cfg = load_yaml(model_config)
    else:
        config_path_str = "inline_dict"
        cfg = dict(model_config)

    data_cfg = {}
    data_yaml_path = CONFIGS_DIR / "data.yaml"
    if data_yaml_path.is_file():
        try:
            data_cfg = load_config("data")
        except Exception as e:
            logger.warning("Không thể đọc configs/data.yaml: %s", e)

    if task is None:
        task = data_cfg.get("task", "multiclass")

    if dedup is None:
        dedup = data_cfg.get("dedup", False)

    variant = "dedup" if dedup else "full"

    if seed is None:
        seed = int(cfg.get("seed", data_cfg.get("split", {}).get("seed", 42)))

    model_key = cfg.get("model") or cfg.get("model_key")
    if not model_key:
        raise ValueError("Cấu hình huấn luyện thiếu trường 'model' hoặc 'model_key'.")

    model_name = cfg.get("name") or model_key
    cfg.setdefault("name", model_name)

    # 2. set_seed
    set_seed(seed)

    # 3. Nạp dữ liệu: chỉ train và val, tuyệt đối include_test=False
    if data is None:
        data = load_experiment_data(
            task=task,
            variant=variant,
            include_test=False,
            max_train_rows=max_train_rows,
        )
    else:
        if getattr(data, "task", None) is not None:
            task = data.task
        if getattr(data, "variant", None) is not None:
            variant = data.variant

    run_name = f"{cfg['name']}_{task}_{variant}{run_suffix}"

    # 4. Trọng số lớp từ y_train nếu imbalance.strategy == 'class_weight'
    imbalance_cfg = cfg.get("imbalance", {})
    sample_weight = None
    class_weight_arr = None

    if (
        isinstance(imbalance_cfg, dict)
        and imbalance_cfg.get("strategy") == "class_weight"
    ):
        power = float(imbalance_cfg.get("power", 0.75))
        class_weight_arr = class_weights(
            data.y_train, n_classes=data.n_classes, power=power
        )
        sample_weight = sample_weights(data.y_train, class_weight_arr)

    # 5. build_model và fit
    params = cfg.get("params", {})
    model = build_model(
        model_key=model_key,
        params=params,
        n_classes=data.n_classes,
        seed=seed,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train=data.X_train,
        y_train=data.y_train,
        X_val=data.X_val,
        y_val=data.y_val,
        sample_weight=sample_weight,
    )
    train_seconds = float(time.perf_counter() - t0)

    # 6. Dự đoán trên validation và tính toàn bộ metrics
    proba_val = model.predict_proba(data.X_val)
    val_metrics = compute_all(
        y_true=data.y_val,
        proba=proba_val,
        class_names=data.class_names,
    )

    # 7. Lưu models/<run>/{model.*, scaler.json, meta.json}
    target_models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    run_dir = target_models_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model.save(run_dir)
    data.scaler.save_json(run_dir / "scaler.json")

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:
        versions["xgboost"] = None

    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = None

    backend = getattr(
        model,
        "backend",
        model.fit_info.get("backend", getattr(model, "name", model_key)),
    )

    meta = {
        "run_name": run_name,
        "model_key": model_key,
        "params": params,
        "task": task,
        "variant": variant,
        "seed": seed,
        "n_classes": data.n_classes,
        "class_names": data.class_names,
        "feature_order": getattr(data.scaler, "feature_names", None),
        "train_rows": int(data.X_train.shape[0]),
        "val_rows": int(data.X_val.shape[0]),
        "imbalance": imbalance_cfg,
        "class_weights": (
            class_weight_arr.tolist() if class_weight_arr is not None else None
        ),
        "backend": backend,
        "versions": versions,
        "train_seconds": train_seconds,
        "fit_info": model.fit_info,
    }

    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 8. Ghi reports/metrics/train_<run>.json (meta + val + fit_info)
    target_metrics_dir = (
        Path(metrics_dir) if metrics_dir is not None else METRICS_DIR
    )
    target_metrics_dir.mkdir(parents=True, exist_ok=True)

    train_metrics_path = target_metrics_dir / f"train_{run_name}.json"
    metrics_payload = {
        "run_name": run_name,
        "meta": meta,
        "val": val_metrics,
        "fit_info": model.fit_info,
    }
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    # 9. Ghi nhật ký vào docs/experiment_log.md nếu bật
    if log_to_experiment_log:
        today_str = datetime.date.today().isoformat()
        val_auc = float(val_metrics.get("roc_auc_ovr_macro", float("nan")))
        val_f1 = float(val_metrics.get("macro_f1", float("nan")))
        note_str = f"train: {task}, {variant}"
        append_experiment_log(
            date=today_str,
            model=model_key,
            config=config_path_str,
            auc=val_auc,
            macro_f1=val_f1,
            note=note_str,
        )

    logger.info(
        "Hoàn tất huấn luyện run '%s' trong %.2f giây (val AUC: %s, val Macro-F1: %s)",
        run_name,
        train_seconds,
        val_metrics.get("roc_auc_ovr_macro"),
        val_metrics.get("macro_f1"),
    )
    return metrics_payload
