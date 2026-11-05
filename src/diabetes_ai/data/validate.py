"""Kiểm tra tính hợp lệ, giá trị ngoài miền và phân phối của dữ liệu BRFSS."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from diabetes_ai.data.schema import FEATURE_RANGES, FEATURES, TARGET
from diabetes_ai.paths import METRICS_DIR


@dataclass
class ValidationReport:
    """Báo cáo kết quả kiểm tra tính hợp lệ của DataFrame dữ liệu."""

    n_rows: int
    n_cols: int
    missing: dict[str, int]
    n_duplicates: int
    n_unique_rows: int
    out_of_range: dict[str, int]
    label_counts: dict[str, int]
    label_pct: dict[str, float]
    feature_stats: dict[str, dict[str, Any]]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        """Kiểm tra toàn bộ điều kiện hợp lệ: không thiếu, không ngoài miền, nhãn ⊆ {0, 1, 2}."""
        no_missing = sum(self.missing.values()) == 0
        no_out_of_range = sum(self.out_of_range.values()) == 0
        valid_labels = set(str(k) for k in self.label_counts.keys()).issubset({"0", "1", "2"})
        return no_missing and no_out_of_range and valid_labels

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi báo cáo thành dict để lưu trữ JSON."""
        data = asdict(self)
        data["ok"] = self.ok
        return data

    def raise_if_invalid(self) -> None:
        """Ném ngoại lệ ValueError nếu dữ liệu không đạt yêu cầu tính hợp lệ."""
        if self.ok:
            return

        errors: list[str] = []
        for col, cnt in self.missing.items():
            if cnt > 0:
                errors.append(f"Cột '{col}' có {cnt} giá trị thiếu (missing/NaN).")

        for col, cnt in self.out_of_range.items():
            if cnt > 0:
                rng = FEATURE_RANGES.get(col)
                rng_info = f" ngoài khoảng cho phép [{rng[0]}, {rng[1]}]" if rng else ""
                errors.append(f"Cột '{col}' có {cnt} dòng{rng_info}.")

        invalid_labels = set(str(k) for k in self.label_counts.keys()) - {"0", "1", "2"}
        if invalid_labels:
            errors.append(
                f"Cột nhãn '{TARGET}' chứa giá trị không hợp lệ: {sorted(invalid_labels)} "
                f"(khoảng cho phép: {{0, 1, 2}})."
            )

        raise ValueError("Dữ liệu không đạt tiêu chuẩn hợp lệ:\n" + "\n".join(errors))


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Kiểm tra toàn diện dữ liệu DataFrame theo các ràng buộc lược đồ.

    Args:
        df: DataFrame cần kiểm tra.

    Returns:
        ValidationReport: Báo cáo chi tiết về chất lượng và độ hợp lệ của dữ liệu.
    """
    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    # Đếm số lượng giá trị thiếu theo cột (chỉ lưu các cột có thiếu)
    missing: dict[str, int] = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        if cnt > 0:
            missing[col] = cnt

    # Số dòng trùng lặp tính trên toàn bộ các cột
    n_duplicates = int(df.duplicated().sum())
    n_unique_rows = n_rows - n_duplicates

    # Kiểm tra giá trị ngoài miền định nghĩa trong FEATURE_RANGES
    out_of_range: dict[str, int] = {}
    for col, (min_val, max_val) in FEATURE_RANGES.items():
        if col in df.columns:
            cnt = int(((df[col] < min_val) | (df[col] > max_val)).sum())
            if cnt > 0:
                out_of_range[col] = cnt

    # Phân phối nhãn mục tiêu
    label_counts: dict[str, int] = {}
    label_pct: dict[str, float] = {}
    if TARGET in df.columns:
        counts = df[TARGET].value_counts().sort_index().to_dict()
        label_counts = {str(int(k)): int(v) for k, v in counts.items()}
        if n_rows > 0:
            label_pct = {k: round(float(v / n_rows), 4) for k, v in label_counts.items()}

    # Thống kê cơ bản min, max, nunique cho từng đặc trưng
    feature_stats: dict[str, dict[str, Any]] = {}
    for col in FEATURES:
        if col in df.columns:
            feature_stats[col] = {
                "min": float(df[col].min()) if n_rows > 0 else 0.0,
                "max": float(df[col].max()) if n_rows > 0 else 0.0,
                "nunique": int(df[col].nunique()),
            }

    # Cảnh báo không chặn (không tính là lỗi)
    warnings: list[str] = []

    # 1. Có dòng trùng (kèm %)
    if n_duplicates > 0:
        pct = (n_duplicates / n_rows * 100.0) if n_rows > 0 else 0.0
        warnings.append(f"Có {n_duplicates} dòng trùng lặp ({pct:.2f}%).")

    # 2. Education=1 dưới 200 dòng
    if "Education" in df.columns:
        edu1_count = int((df["Education"] == 1).sum())
        if edu1_count < 200:
            warnings.append(f"Cột Education=1 có {edu1_count} dòng (< 200 dòng).")

    # 3. BMI > 60
    if "BMI" in df.columns:
        bmi_gt_60 = int((df["BMI"] > 60).sum())
        if bmi_gt_60 > 0:
            warnings.append(f"Cột BMI có {bmi_gt_60} dòng > 60.")

    # 4. Tỉ lệ lớp hiếm nhất < 2%
    if label_counts and n_rows > 0:
        min_count = min(label_counts.values())
        min_ratio = min_count / n_rows
        if min_ratio < 0.02:
            warnings.append(f"Tỉ lệ lớp hiếm nhất < 2% ({min_ratio * 100.0:.2f}%).")

    return ValidationReport(
        n_rows=n_rows,
        n_cols=n_cols,
        missing=missing,
        n_duplicates=n_duplicates,
        n_unique_rows=n_unique_rows,
        out_of_range=out_of_range,
        label_counts=label_counts,
        label_pct=label_pct,
        feature_stats=feature_stats,
        warnings=warnings,
    )


def write_report(
    report: ValidationReport,
    path: Path | str | None = None,
) -> Path:
    """Ghi báo cáo kiểm tra dữ liệu ra file JSON.

    Args:
        report: Đối tượng ValidationReport cần ghi.
        path: Đường dẫn file JSON. Mặc định là reports/metrics/data_validation.json.

    Returns:
        Path: Đường dẫn tới file JSON đã ghi.
    """
    if path is None:
        target_path = METRICS_DIR / "data_validation.json"
    else:
        target_path = Path(path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    return target_path
