"""Script chạy giải thích toàn cục: tính độ quan trọng hoán vị trên tập validation và vẽ what-if."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Đảm bảo import được gói src
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diabetes_ai.config import load_config
from diabetes_ai.data.schema import FEATURES
from diabetes_ai.evaluation.plots import plot_importance, plot_what_if
from diabetes_ai.explain.permutation import permutation_importance
from diabetes_ai.explain.what_if import what_if_scenarios
from diabetes_ai.features.build_features import load_experiment_data
from diabetes_ai.models.registry import load_run


def parse_args() -> argparse.Namespace:
    """Xử lý các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Tính độ quan trọng hoán vị toàn cục trên tập validation và vẽ kịch bản what-if."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "binary"],
        help="Tác vụ phân loại (mặc định: multiclass).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["full", "dedup"],
        help="Biến thể tập dữ liệu (mặc định: full).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Tên run mô hình (mặc định lấy mô hình tốt nhất từ model_selection.json).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Số lần xáo trộn ngẫu nhiên độc lập cho mỗi đặc trưng (mặc định: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên để tái lập kết quả (mặc định: 42).",
    )
    return parser.parse_args()


def get_default_run(task: str, variant: str) -> str:
    """Lấy run tốt nhất từ file reports/metrics/model_selection.json."""
    selection_file = ROOT / "reports" / "metrics" / "model_selection.json"
    if not selection_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file lựa chọn mô hình: {selection_file}")

    with open(selection_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = f"{task}_{variant}"
    if key not in data or "best" not in data[key]:
        raise KeyError(f"Không tìm thấy cấu hình best model cho khóa '{key}' trong {selection_file}")

    return str(data[key]["best"])


def main() -> None:
    """Hàm thực thi chính."""
    args = parse_args()

    run_name = args.run if args.run else get_default_run(args.task, args.variant)
    print(f"=== BẮT ĐẦU GIẢI THÍCH TOÀN CỤC CHO RUN: {run_name} ===")
    print(f"Tác vụ: {args.task}, Biến thể: {args.variant}, Số lần lặp hoán vị: {args.n_repeats}, Seed: {args.seed}")

    # 1. Nạp dữ liệu trên tập validation (include_test luôn là False)
    print("Nạp dữ liệu thực nghiệm trên tập validation (include_test=False)...")
    data = load_experiment_data(
        task=args.task,
        variant=args.variant,
        include_test=False,
    )

    # 2. Nạp mô hình
    print(f"Nạp mô hình đã huấn luyện: {run_name}...")
    run = load_run(run_name)

    # 3. Tính độ quan trọng hoán vị trên validation
    print(f"Đang tính permutation importance trên {len(data.X_val)} mẫu validation...")
    predict_fn = lambda Xs: run.predict_proba_scaled(Xs, calibrated=True)

    importance_rows = permutation_importance(
        predict_fn=predict_fn,
        X=data.X_val,
        y=data.y_val,
        features=list(FEATURES),
        n_repeats=args.n_repeats,
        seed=args.seed,
        metric="roc_auc_ovr",
    )

    # Ghi kết quả hoán vị vào JSON (21 mục)
    metrics_dir = ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"permutation_importance_{run_name}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(importance_rows, f, ensure_ascii=False, indent=2)
    print(f"-> Đã ghi: {json_path} ({len(importance_rows)} đặc trưng)")

    # 4. Vẽ biểu đồ độ quan trọng hoán vị
    figures_dir = ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_importance_path = figures_dir / f"permutation_importance_{run_name}.png"
    plot_importance(importance_rows, fig_importance_path, top=15)
    print(f"-> Đã lưu biểu đồ hoán vị: {fig_importance_path}")

    # 5. Kịch bản what-if cho hồ sơ mẫu sample_profile.json
    sample_path = ROOT / "tests" / "fixtures" / "sample_profile.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_profile = json.load(f)

        features_cfg = load_config("features")
        x_raw = np.array([float(sample_profile[feat]) for feat in FEATURES], dtype=float)

        def index_fn(X_mat: np.ndarray) -> np.ndarray:
            mat_2d = np.atleast_2d(X_mat)
            proba = run.predict_proba(mat_2d, calibrated=True)
            if args.task == "multiclass":
                return proba[:, 2] + 0.5 * proba[:, 1]
            col = 1 if proba.shape[1] > 1 else 0
            return proba[:, col]

        scenarios = what_if_scenarios(index_fn, x_raw, cfg=features_cfg)
        fig_what_if_path = figures_dir / f"what_if_sample_{run_name}.png"
        plot_what_if(scenarios, fig_what_if_path)
        print(f"-> Đã tạo {len(scenarios)} kịch bản what-if cho sample_profile và lưu: {fig_what_if_path}")

    top5 = [x["feature"] for x in importance_rows[:5]]
    print(f"Top 5 đặc trưng quan trọng nhất: {top5}")
    print("=== HOÀN TẤT GIẢI THÍCH TOÀN CỤC ===")


if __name__ == "__main__":
    main()
