"""Nạp và kiểm tra sơ bộ dữ liệu BRFSS thô từ file CSV."""

from pathlib import Path
from typing import Any

import pandas as pd

from diabetes_ai.config import load_config
from diabetes_ai.data.schema import FEATURES, TARGET
from diabetes_ai.paths import ROOT


def load_raw(
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Nạp file CSV dữ liệu thô BRFSS và chuẩn hóa các cột.

    Args:
        path: Đường dẫn tới file CSV thô. Nếu None, lấy theo config.
        config: Dict cấu hình data. Nếu None, đọc từ configs/data.yaml.

    Returns:
        pd.DataFrame: DataFrame gồm TARGET (int64) và FEATURES (float32).

    Raises:
        FileNotFoundError: Nếu file CSV không tồn tại.
        ValueError: Nếu thiếu cột, có cột lạ hoặc có giá trị không nguyên.
    """
    if path is None:
        if config is None:
            config = load_config("data")
        raw_path = config.get("raw_path", "data/raw/brfss2015/diabetes_012_health_indicators_BRFSS2015.csv")
        file_path = ROOT / raw_path
        display_path = str(file_path)
    else:
        file_path = Path(path)
        display_path = str(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại '{display_path}'. "
            f"Đặt CSV vào data/raw/brfss2015/ (xem data/README.md)."
        )

    df = pd.read_csv(file_path, encoding="utf-8")

    expected_cols = [TARGET] + FEATURES
    expected_set = set(expected_cols)
    actual_set = set(df.columns)

    missing = expected_set - actual_set
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}")

    extra = actual_set - expected_set
    if extra:
        raise ValueError(f"Cột lạ: {sorted(extra)}")

    # Sắp xếp các cột theo thứ tự chuẩn [TARGET] + FEATURES
    df = df[expected_cols].copy()

    # Kiểm tra mọi giá trị là số nguyên
    for col in expected_cols:
        bad = (df[col] % 1 != 0).sum()
        if bad:
            raise ValueError(f"Cột {col} có {bad} giá trị không nguyên")

    # Ép kiểu dữ liệu
    df[TARGET] = df[TARGET].astype("int64")
    for col in FEATURES:
        df[col] = df[col].astype("float32")

    return df.reset_index(drop=True)
