"""Kiểm thử tiền xử lý dữ liệu và nạp dữ liệu thô."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diabetes_ai.data.load import load_raw
from diabetes_ai.data.preprocess import (
    drop_exact_duplicates,
    duplicate_mask,
    encode_target,
)
from diabetes_ai.data.schema import FEATURES, TARGET
from diabetes_ai.data.validate import validate_dataframe


def test_load_raw_returns_expected_columns_and_dtypes(tmp_path: Path):
    """load_raw nạp đúng cấu trúc cột và ép kiểu TARGET int64, FEATURES float32."""
    cols = [TARGET] + FEATURES
    fake_data = {col: [0, 1] for col in cols}
    fake_df = pd.DataFrame(fake_data)
    csv_file = tmp_path / "fake_data.csv"
    fake_df.to_csv(csv_file, index=False)

    loaded = load_raw(csv_file)
    assert list(loaded.columns) == cols
    assert loaded[TARGET].dtype == np.int64
    for col in FEATURES:
        assert loaded[col].dtype == np.float32
    assert len(loaded) == 2


def test_load_raw_missing_file_message(tmp_path: Path):
    """load_raw ném FileNotFoundError kèm thông báo tiếng Việt khi file không tồn tại."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Không tìm thấy dữ liệu"):
        load_raw(non_existent)


def test_load_raw_missing_column_raises(tmp_path: Path):
    """load_raw ném ValueError với thông báo 'Thiếu cột' khi CSV thiếu cột theo schema."""
    cols = [TARGET] + FEATURES[:-1]  # Thiếu cột cuối cùng
    fake_data = {col: [0, 1] for col in cols}
    fake_df = pd.DataFrame(fake_data)
    csv_file = tmp_path / "incomplete.csv"
    fake_df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Thiếu cột"):
        load_raw(csv_file)


def test_validate_flags_out_of_range_and_missing():
    """validate_dataframe phát hiện ô thiếu và giá trị ngoài miền; raise_if_invalid ném lỗi."""
    data = {TARGET: [0, 1, 2], "BMI": [25.0, np.nan, 120.0]}  # 120 > max 100
    for col in FEATURES:
        if col != "BMI":
            data[col] = [0.0, 1.0, 0.0]

    df = pd.DataFrame(data)
    report = validate_dataframe(df)

    assert not report.ok
    assert "BMI" in report.missing
    assert report.missing["BMI"] == 1
    assert "BMI" in report.out_of_range
    assert report.out_of_range["BMI"] == 1

    with pytest.raises(ValueError, match="Dữ liệu không đạt tiêu chuẩn"):
        report.raise_if_invalid()


def test_validate_counts_duplicates():
    """validate_dataframe đếm chính xác số dòng trùng lặp và số dòng duy nhất."""
    cols = [TARGET] + FEATURES
    row = {col: 0.0 for col in cols}
    row[TARGET] = 0
    # 4 dòng: 3 dòng giống hệt nhau, 1 dòng khác
    row_alt = dict(row)
    row_alt["BMI"] = 35.0

    df = pd.DataFrame([row, row, row, row_alt])
    report = validate_dataframe(df)

    assert report.n_rows == 4
    assert report.n_duplicates == 2
    assert report.n_unique_rows == 2


def test_encode_target_binary_merges_classes():
    """encode_target gộp nhãn 1 và 2 thành 1 cho binary; multiclass giữ nguyên."""
    y = np.array([0, 1, 2], dtype=np.int64)

    y_multi = encode_target(y, task="multiclass")
    np.testing.assert_array_equal(y_multi, np.array([0, 1, 2], dtype=np.int64))

    y_bin = encode_target(y, task="binary")
    np.testing.assert_array_equal(y_bin, np.array([0, 1, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="task phải là 'multiclass' hoặc 'binary'"):
        encode_target(y, task="unknown_task")


def test_drop_exact_duplicates_keeps_first_and_index():
    """drop_exact_duplicates loại bỏ dòng trùng, giữ bản sao đầu tiên và giữ nguyên index gốc."""
    cols = [TARGET] + FEATURES
    r0 = {col: 1.0 for col in cols}
    r0[TARGET] = 0
    r1 = {col: 2.0 for col in cols}
    r1[TARGET] = 1

    # Tạo df với index tùy chỉnh [10, 20, 30]
    df = pd.DataFrame([r0, r1, r0], index=[10, 20, 30])
    mask = duplicate_mask(df)
    np.testing.assert_array_equal(mask, [False, False, True])

    deduped = drop_exact_duplicates(df)
    assert len(deduped) == 2
    assert list(deduped.index) == [10, 20]


@pytest.mark.real_data
def test_real_csv_validation_numbers(real_csv_path: Path):
    """Kiểm chứng số liệu chuẩn trên file CSV thật BRFSS 2015."""
    df = load_raw(real_csv_path)
    report = validate_dataframe(df)

    assert report.n_rows == 253680
    assert report.n_duplicates == 23899
    assert report.n_unique_rows == 229781
    assert sum(report.missing.values()) == 0
    assert sum(report.out_of_range.values()) == 0
    assert report.label_counts == {"0": 213703, "1": 4631, "2": 35346}
    assert report.ok is True
