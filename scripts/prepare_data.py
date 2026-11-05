"""Chạy tiền xử lý và chia tập dữ liệu BRFSS, lưu trữ chỉ số và manifest."""

import argparse
import sys
from pathlib import Path

# Đảm bảo import được src/ khi chạy script trực tiếp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diabetes_ai.config import load_config
from diabetes_ai.data.load import load_raw
from diabetes_ai.data.split import build_manifest, make_split, save_split
from diabetes_ai.data.validate import validate_dataframe, write_report
from diabetes_ai.paths import ROOT


def main() -> int:
    """Hàm chính thực thi kiểm tra, phân chia dữ liệu và lưu manifest."""
    parser = argparse.ArgumentParser(
        description="Nạp dữ liệu BRFSS, kiểm tra hợp lệ, chia tập và xuất chỉ số."
    )
    parser.add_argument(
        "--only",
        choices=["full", "dedup"],
        default=None,
        help="Chỉ tạo split cho một biến thể cụ thể ('full' hoặc 'dedup'). Mặc định tạo cả hai.",
    )
    args = parser.parse_args()

    cfg = load_config("data")
    raw_path = cfg.get(
        "raw_path",
        "data/raw/brfss2015/diabetes_012_health_indicators_BRFSS2015.csv",
    )
    split_cfg = cfg.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.15))
    seed = int(split_cfg.get("seed", 42))
    splits_dir = Path(cfg.get("splits_dir", "data/splits"))
    if not splits_dir.is_absolute():
        splits_dir = ROOT / splits_dir

    csv_file = ROOT / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
    df = load_raw(csv_file)

    report = validate_dataframe(df)
    report.raise_if_invalid()
    write_report(report)

    variants = [args.only] if args.only else ["full", "dedup"]

    for variant in variants:
        dedup = (variant == "dedup")
        split = make_split(
            df=df,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            dedup=dedup,
        )
        manifest = build_manifest(
            df=df,
            split=split,
            test_size=test_size,
            val_size=val_size,
        )
        save_split(split, manifest, splits_dir=splits_dir)

        total = len(split.train) + len(split.val) + len(split.test)
        val_twins_n = manifest["twins"]["val"]["n"]
        val_pct = (val_twins_n / len(split.val) * 100.0) if len(split.val) > 0 else 0.0
        test_twins_n = manifest["twins"]["test"]["n"]
        test_pct = (test_twins_n / len(split.test) * 100.0) if len(split.test) > 0 else 0.0

        print(
            f"[{variant}] train={len(split.train)} val={len(split.val)} test={len(split.test)} total={total}"
        )
        print(
            f"[{variant}] val_twins={val_twins_n} ({val_pct:.2f}%) test_twins={test_twins_n} ({test_pct:.2f}%)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
