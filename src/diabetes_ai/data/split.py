"""Chia dữ liệu train/val/test phân tầng, kiểm tra rò rỉ và lưu trữ chỉ số."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_ai.data.preprocess import duplicate_mask
from diabetes_ai.data.schema import FEATURES, TARGET
from diabetes_ai.paths import SPLITS_DIR


@dataclass(frozen=True)
class SplitIndices:
    """Chứa chỉ số (index) dòng cho các tập train, val, test."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    variant: str
    seed: int

    def sizes(self) -> dict[str, int]:
        """Trả về kích thước từng tập."""
        return {
            "train": int(len(self.train)),
            "val": int(len(self.val)),
            "test": int(len(self.test)),
        }


def variant_name(dedup: bool) -> str:
    """Trả về tên biến thể dữ liệu ('dedup' hoặc 'full')."""
    return "dedup" if dedup else "full"


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
    dedup: bool = False,
) -> SplitIndices:
    """Chia DataFrame thành các tập train, val, test phân tầng theo nhãn mục tiêu.

    Chỉ số trả về là số hàng của DataFrame gốc.

    Args:
        df: DataFrame đầu vào. Yêu cầu df.index là RangeIndex.
        test_size: Tỉ lệ tập test so với toàn bộ dữ liệu sử dụng.
        val_size: Tỉ lệ tập val so với tập trainval.
        seed: Seed sinh số ngẫu nhiên.
        dedup: Có loại bỏ dòng trùng lặp trước khi chia hay không.

    Returns:
        SplitIndices: Đối tượng chứa các mảng chỉ số đã được sắp xếp tăng dần.

    Raises:
        ValueError: Nếu df.index không phải RangeIndex hoặc nếu phát hiện rò rỉ chỉ số.
    """
    if not isinstance(df.index, pd.RangeIndex):
        raise ValueError(
            f"Chỉ số của DataFrame (df.index) phải là pd.RangeIndex, nhận được: {type(df.index)}."
        )

    ids = df.index.to_numpy()
    if dedup:
        ids = ids[~duplicate_mask(df)]

    y = df.loc[ids, TARGET].to_numpy()
    trainval, test = train_test_split(
        ids,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    y_tv = df.loc[trainval, TARGET].to_numpy()
    train, val = train_test_split(
        trainval,
        test_size=val_size,
        stratify=y_tv,
        random_state=seed,
    )

    train = np.sort(train).astype(np.int64)
    val = np.sort(val).astype(np.int64)
    test = np.sort(test).astype(np.int64)

    # Kiểm tra tính toàn vẹn và chống rò rỉ chỉ số
    s_train = set(train.tolist())
    s_val = set(val.tolist())
    s_test = set(test.tolist())

    if s_train & s_val or s_train & s_test or s_val & s_test:
        raise ValueError("Rò rỉ chỉ số giữa các tập: các tập train, val, test không rời nhau.")

    if s_train | s_val | s_test != set(ids.tolist()):
        raise ValueError(
            "Rò rỉ chỉ số giữa các tập: hợp của train, val, test không bằng tập chỉ số đầu vào."
        )

    if len(train) + len(val) + len(test) != len(ids):
        raise ValueError(
            "Rò rỉ chỉ số giữa các tập: tổng kích thước các tập không bằng số lượng chỉ số ban đầu."
        )

    return SplitIndices(
        train=train,
        val=val,
        test=test,
        variant=variant_name(dedup),
        seed=seed,
    )


def twin_mask(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
) -> np.ndarray:
    """Xác định các dòng trong query_idx có vector đặc trưng (21 cột) giống hệt dòng nào đó trong train_idx.

    Args:
        df: DataFrame chứa đầy đủ các cột FEATURES.
        train_idx: Chỉ số các dòng thuộc tập train.
        query_idx: Chỉ số các dòng cần truy vấn (thường là val hoặc test).

    Returns:
        np.ndarray: Mảng boolean tương ứng với query_idx, True nếu dòng đó là bản sinh đôi (twin).
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)

    if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
        h = pd.util.hash_pandas_object(df[FEATURES].astype("int64"), index=False).to_numpy()
        return np.isin(h[query_idx], h[train_idx])

    h_series = pd.util.hash_pandas_object(df[FEATURES].astype("int64"), index=False)
    return np.isin(h_series.loc[query_idx].to_numpy(), h_series.loc[train_idx].to_numpy())


def build_manifest(
    df: pd.DataFrame,
    split: SplitIndices,
    test_size: float = 0.2,
    val_size: float = 0.15,
) -> dict[str, Any]:
    """Xây dựng thông tin kiểm kê (manifest) chi tiết cho một phép chia dữ liệu.

    Args:
        df: DataFrame gốc ban đầu.
        split: Đối tượng SplitIndices đã được chia.
        test_size: Tỉ lệ tập test đã sử dụng.
        val_size: Tỉ lệ tập val đã sử dụng.

    Returns:
        dict[str, Any]: Dict chứa toàn bộ thông số phân tách, sha256 và thống kê bản sinh đôi.
    """
    unique_classes = sorted([int(c) for c in np.unique(df[TARGET])])

    class_counts: dict[str, dict[str, int]] = {}
    class_pct: dict[str, dict[str, float]] = {}

    for name, idx in [("train", split.train), ("val", split.val), ("test", split.test)]:
        y_sub = df.loc[idx, TARGET].to_numpy()
        n_sub = len(idx)
        counts = {str(c): int(np.sum(y_sub == c)) for c in unique_classes}
        pcts = {
            str(c): round(float(counts[str(c)] / n_sub), 4) if n_sub > 0 else 0.0
            for c in unique_classes
        }
        class_counts[name] = counts
        class_pct[name] = pcts

    sha256_dict = {
        "train": hashlib.sha256(split.train.astype("int64").tobytes()).hexdigest(),
        "val": hashlib.sha256(split.val.astype("int64").tobytes()).hexdigest(),
        "test": hashlib.sha256(split.test.astype("int64").tobytes()).hexdigest(),
    }

    val_twins_mask = twin_mask(df, split.train, split.val)
    test_twins_mask = twin_mask(df, split.train, split.test)
    val_twins_n = int(np.sum(val_twins_mask))
    test_twins_n = int(np.sum(test_twins_mask))

    val_len = len(split.val)
    test_len = len(split.test)

    twins_dict = {
        "val": {
            "n": val_twins_n,
            "pct": round(float(val_twins_n / val_len), 4) if val_len > 0 else 0.0,
        },
        "test": {
            "n": test_twins_n,
            "pct": round(float(test_twins_n / test_len), 4) if test_len > 0 else 0.0,
        },
    }

    return {
        "variant": split.variant,
        "seed": split.seed,
        "test_size": float(test_size),
        "val_size": float(val_size),
        "stratify": True,
        "dedup": split.variant == "dedup",
        "n_rows_raw": int(len(df)),
        "n_rows_used": int(len(split.train) + len(split.val) + len(split.test)),
        "sizes": split.sizes(),
        "class_counts": class_counts,
        "class_pct": class_pct,
        "sha256": sha256_dict,
        "twins": twins_dict,
    }


def save_split(
    split: SplitIndices,
    manifest: dict[str, Any],
    splits_dir: Path | str | None = None,
) -> Path:
    """Lưu mảng chỉ số (indices.npz) và manifest.json vào thư mục đích.

    Args:
        split: Đối tượng SplitIndices cần lưu.
        manifest: Dữ liệu manifest tương ứng.
        splits_dir: Thư mục chứa các split. Mặc định là data/splits/.

    Returns:
        Path: Đường dẫn tới thư mục con chứa kết quả lưu trữ.
    """
    if splits_dir is None:
        target_dir = SPLITS_DIR / f"{split.variant}_seed{split.seed}"
    else:
        target_dir = Path(splits_dir) / f"{split.variant}_seed{split.seed}"

    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_dir / "indices.npz",
        train=split.train.astype(np.int64),
        val=split.val.astype(np.int64),
        test=split.test.astype(np.int64),
    )
    with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return target_dir


def load_split(
    variant: str = "full",
    seed: int = 42,
    splits_dir: Path | str | None = None,
) -> SplitIndices:
    """Nạp các chỉ số phân chia dữ liệu từ file lưu trữ.

    Args:
        variant: Tên biến thể ('full' hoặc 'dedup').
        seed: Seed dùng khi chia.
        splits_dir: Thư mục chứa các split. Mặc định là data/splits/.

    Returns:
        SplitIndices: Đối tượng chứa mảng chỉ số nạp từ đĩa.

    Raises:
        FileNotFoundError: Nếu file indices.npz chưa tồn tại.
    """
    if splits_dir is None:
        target_dir = SPLITS_DIR / f"{variant}_seed{seed}"
    else:
        target_dir = Path(splits_dir) / f"{variant}_seed{seed}"

    npz_path = target_dir / "indices.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Chưa có split '{variant}' với seed {seed} tại '{target_dir}'. "
            "Hãy chạy: python scripts/prepare_data.py"
        )

    with np.load(npz_path) as data:
        train = data["train"].astype(np.int64)
        val = data["val"].astype(np.int64)
        test = data["test"].astype(np.int64)

    return SplitIndices(
        train=train,
        val=val,
        test=test,
        variant=variant,
        seed=seed,
    )
