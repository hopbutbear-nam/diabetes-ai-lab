"""Kịch bản xuất mô hình tốt nhất thành gói bundle tự đủ phục vụ suy luận."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from diabetes_ai.inference.export import export_best, load_bundle
from diabetes_ai.paths import METRICS_DIR, ROOT


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Xuất mô hình tốt nhất theo model_selection.json thành gói bundle tự đủ."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "binary"],
        help="Tác vụ phân loại (mặc định: 'multiclass').",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=["full", "dedup"],
        help="Biến thể dữ liệu (mặc định: 'full').",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Đường dẫn lưu file bundle (mặc định: models/best_model.joblib).",
    )
    return parser.parse_args()


def main() -> None:
    """Thực thi xuất mô hình và ghi tóm tắt metrics."""
    args = parse_args()
    out_path = Path(args.out) if args.out else ROOT / "models" / "best_model.joblib"

    saved_path = export_best(task=args.task, variant=args.variant, out=out_path)
    bundle = load_bundle(saved_path)

    file_bytes = os.path.getsize(saved_path)
    with open(saved_path, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()

    run_name = bundle.get("run_name", "unknown")
    task = bundle.get("task", args.task)
    size_mb = round(file_bytes / 1e6, 2)

    print(
        f"Xuất mô hình thành công: run='{run_name}', task='{task}' -> {saved_path} "
        f"({size_mb} MB, {file_bytes} bytes, sha256={file_sha256[:12]}...)"
    )

    summary = {
        "run_name": run_name,
        "task": task,
        "variant": args.variant,
        "path": str(saved_path),
        "bytes": file_bytes,
        "sha256": file_sha256,
    }

    summary_path = METRICS_DIR / "export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Đã ghi báo cáo tóm tắt xuất mô hình -> {summary_path}")


if __name__ == "__main__":
    main()
