"""Kiểm thử tính toàn vẹn của phép chia dữ liệu và cơ chế chống rò rỉ."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diabetes_ai.data.load import load_raw
from diabetes_ai.data.schema import FEATURES, TARGET
from diabetes_ai.data.split import (
    build_manifest,
    load_split,
    make_split,
    save_split,
    twin_mask,
)


def test_split_disjoint_and_cover(synthetic_df: pd.DataFrame):
    """Các tập train, val, test phải rời nhau từng đôi một và phủ kín toàn bộ chỉ số."""
    split = make_split(synthetic_df, test_size=0.2, val_size=0.15, seed=42)

    s_train = set(split.train.tolist())
    s_val = set(split.val.tolist())
    s_test = set(split.test.tolist())

    assert len(s_train & s_val) == 0
    assert len(s_train & s_test) == 0
    assert len(s_val & s_test) == 0

    assert (s_train | s_val | s_test) == set(synthetic_df.index.tolist())
    assert len(split.train) + len(split.val) + len(split.test) == len(synthetic_df)


def test_split_is_stratified(synthetic_df: pd.DataFrame):
    """Tỉ lệ từng lớp mục tiêu lệch không quá 0.5 điểm % (0.005) giữa các tập."""
    split = make_split(synthetic_df, test_size=0.2, val_size=0.15, seed=42)

    for c in [0, 1, 2]:
        p_train = np.mean(synthetic_df.loc[split.train, TARGET] == c)
        p_val = np.mean(synthetic_df.loc[split.val, TARGET] == c)
        p_test = np.mean(synthetic_df.loc[split.test, TARGET] == c)

        assert abs(p_train - p_val) <= 0.005
        assert abs(p_train - p_test) <= 0.005
        assert abs(p_val - p_test) <= 0.005


def test_split_deterministic_for_same_seed(synthetic_df: pd.DataFrame):
    """Hai lần gọi make_split với cùng seed trả về các mảng chỉ số giống hệt nhau."""
    split1 = make_split(synthetic_df, seed=42)
    split2 = make_split(synthetic_df, seed=42)

    np.testing.assert_array_equal(split1.train, split2.train)
    np.testing.assert_array_equal(split1.val, split2.val)
    np.testing.assert_array_equal(split1.test, split2.test)


def test_split_dedup_has_no_exact_duplicates(synthetic_df: pd.DataFrame):
    """Khi dedup=True, toàn bộ các dòng được giữ lại trong 3 tập không chứa bản ghi trùng."""
    split = make_split(synthetic_df, dedup=True, seed=42)
    all_indices = np.concatenate([split.train, split.val, split.test])

    subset_df = synthetic_df.loc[all_indices]
    assert subset_df.duplicated().sum() == 0


def test_twin_mask_counts():
    """twin_mask đếm chính xác số bản sinh đôi và khớp với đối chiếu set tuple."""
    cols = [TARGET] + FEATURES
    row_a = {col: 1.0 for col in cols}
    row_a[TARGET] = 0
    row_b = {col: 2.0 for col in cols}
    row_b[TARGET] = 1
    row_c = {col: 3.0 for col in cols}
    row_c[TARGET] = 2

    # Tập train gồm row_a và row_b
    # Tập query gồm row_a (twin), row_c (không twin), row_b (twin)
    df = pd.DataFrame([row_a, row_b, row_a, row_c, row_b])
    train_idx = np.array([0, 1], dtype=np.int64)
    query_idx = np.array([2, 3, 4], dtype=np.int64)

    mask = twin_mask(df, train_idx, query_idx)
    np.testing.assert_array_equal(mask, [True, False, True])
    assert int(mask.sum()) == 2

    # Đối chiếu bằng set các tuple đặc trưng
    train_tuples = {tuple(x) for x in df.loc[train_idx, FEATURES].to_numpy()}
    expected = np.array([tuple(x) in train_tuples for x in df.loc[query_idx, FEATURES].to_numpy()])
    np.testing.assert_array_equal(mask, expected)


def test_save_load_roundtrip_and_hash_in_manifest(
    synthetic_df: pd.DataFrame,
    tmp_path: Path,
):
    """save_split và load_split bảo toàn chỉ số và mã sha256 trong manifest."""
    split = make_split(synthetic_df, seed=42)
    manifest = build_manifest(synthetic_df, split)

    out_dir = save_split(split, manifest, splits_dir=tmp_path)
    assert (out_dir / "indices.npz").is_file()
    assert (out_dir / "manifest.json").is_file()

    loaded_split = load_split(variant=split.variant, seed=split.seed, splits_dir=tmp_path)
    np.testing.assert_array_equal(loaded_split.train, split.train)
    np.testing.assert_array_equal(loaded_split.val, split.val)
    np.testing.assert_array_equal(loaded_split.test, split.test)

    # Kiểm tra sha256 trong manifest
    expected_train_sha = hashlib.sha256(loaded_split.train.astype("int64").tobytes()).hexdigest()
    assert manifest["sha256"]["train"] == expected_train_sha


@pytest.mark.real_data
def test_real_split_sizes_and_class_share(real_csv_path: Path):
    """Kiểm tra kích thước chuẩn và tỉ lệ lớp 2 trên dữ liệu thật BRFSS."""
    df = load_raw(real_csv_path)
    split = make_split(df, test_size=0.2, val_size=0.15, seed=42, dedup=False)

    assert split.sizes() == {
        "train": 172502,
        "val": 30442,
        "test": 50736,
    }

    # Tỉ lệ lớp 2 nằm trong khoảng 13.933% ± 0.3 điểm % (0.13633 đến 0.14233)
    for idx in [split.train, split.val, split.test]:
        share = float(np.mean(df.loc[idx, TARGET] == 2))
        assert abs(share - 0.13933) <= 0.003


@pytest.mark.real_data
def test_real_split_deterministic_hashes(real_csv_path: Path):
    """Hai lần phân chia dữ liệu thật với cùng seed cho ra mã sha256 giống hệt nhau."""
    df = load_raw(real_csv_path)
    split1 = make_split(df, seed=42)
    manifest1 = build_manifest(df, split1)

    split2 = make_split(df, seed=42)
    manifest2 = build_manifest(df, split2)

    assert manifest1["sha256"] == manifest2["sha256"]
