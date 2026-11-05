"""Script huấn luyện mô hình dự đoán tiểu đường từ cấu hình YAML.

Ví dụ sử dụng:
    python scripts/train.py --config configs/train_xgb.yaml
    python scripts/train.py --config configs/train_xgb.yaml --task binary --no-log
"""

from __future__ import annotations

import argparse
import json
import sys

from diabetes_ai.training.train import run_training


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình dự đoán tiểu đường từ file cấu hình YAML."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Đường dẫn tới file cấu hình YAML (ví dụ configs/train_xgb.yaml).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["multiclass", "binary"],
        default=None,
        help="Bài toán dự đoán: 'multiclass' hoặc 'binary'. Mặc định đọc từ configs/data.yaml.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=None,
        help="Sử dụng tập dữ liệu đã khử trùng lặp (dedup).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Hạt giống ngẫu nhiên (seed).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Số dòng huấn luyện tối đa (dùng để gỡ lỗi hoặc chạy thử nghiệm nhanh).",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Hậu tố bổ sung vào tên run (ví dụ _repeat).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Không ghi dòng kết quả vào docs/experiment_log.md.",
    )

    args = parser.parse_args()

    res = run_training(
        model_config=args.config,
        task=args.task,
        dedup=args.dedup,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        run_suffix=args.run_suffix,
        log_to_experiment_log=not args.no_log,
    )

    summary = {
        "run": res["run_name"],
        "val_auc": res["val"].get("roc_auc_ovr_macro"),
        "val_macro_f1": res["val"].get("macro_f1"),
        "train_seconds": res["meta"].get("train_seconds"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
