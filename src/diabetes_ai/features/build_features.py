"""Tạo ma trận X, y từ dữ liệu đã làm sạch và xây dựng ExperimentData."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diabetes_ai.config import load_config
from diabetes_ai.data.load import load_raw
from diabetes_ai.data.preprocess import encode_target
from diabetes_ai.data.schema import FEATURES, TARGET, class_names
from diabetes_ai.data.split import SplitIndices, load_split
from diabetes_ai.features.scalers import StandardScalerNP
from diabetes_ai.logging_utils import get_logger
from diabetes_ai.training.imbalance import subsample_stratified

logger = get_logger(__name__)


def to_xy(
    df: pd.DataFrame,
    idx: np.ndarray,
    task: str = "multiclass",
) -> tuple[np.ndarray, np.ndarray]:
    """Trích xuất ma trận đặc trưng X và vector nhãn y từ DataFrame.

    Args:
        df: DataFrame BRFSS đã làm sạch.
        idx: Mảng chỉ số dòng cần trích xuất.
        task: Tác vụ phân loại ('multiclass' hoặc 'binary').

    Returns:
        tuple: (X float32 shape (n, 21) theo thứ tự FEATURES, y int64)
    """
    idx = np.asarray(idx, dtype=np.int64)
    X = df.loc[idx, FEATURES].to_numpy(dtype=np.float32)
    y_raw = df.loc[idx, TARGET].to_numpy()
    y = encode_target(y_raw, task=task)
    return X, y


@dataclass
class ExperimentData:
    """Chứa toàn bộ dữ liệu thực nghiệm đã chuẩn hóa.

    Attributes:
        X_train: Ma trận đặc trưng train đã chuẩn hóa (float32).
        y_train: Nhãn train (int64).
        X_val: Ma trận đặc trưng val đã chuẩn hóa (float32).
        y_val: Nhãn val (int64).
        X_test: Ma trận đặc trưng test đã chuẩn hóa, None nếu include_test=False.
        y_test: Nhãn test, None nếu include_test=False.
        scaler: StandardScalerNP đã fit trên train.
        X_train_raw: Ma trận đặc trưng train chưa chuẩn hóa (float32).
        pop_mean_raw: Trung bình từng cột trên train (float32, dùng cho what-if).
        idx_train: Chỉ số dòng tập train.
        idx_val: Chỉ số dòng tập val.
        idx_test: Chỉ số dòng tập test.
        task: Tác vụ phân loại.
        variant: Biến thể dữ liệu ('full' hoặc 'dedup').
        class_names: Danh sách tên các lớp.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray | None
    y_test: np.ndarray | None
    scaler: StandardScalerNP
    X_train_raw: np.ndarray
    pop_mean_raw: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    task: str
    variant: str
    class_names: list[str]

    @property
    def n_classes(self) -> int:
        """Số lượng lớp."""
        return len(self.class_names)


def build_experiment_data_from_frame(
    df: pd.DataFrame,
    split: SplitIndices,
    task: str = "multiclass",
    include_test: bool = False,
    max_train_rows: int | None = None,
    seed: int = 42,
) -> ExperimentData:
    """Xây dựng ExperimentData từ DataFrame và SplitIndices.

    Scaler chỉ fit trên tập train (sau lấy mẫu con nếu có).
    Val và test chỉ dùng transform.

    Args:
        df: DataFrame BRFSS đã làm sạch.
        split: SplitIndices chứa chỉ số train/val/test.
        task: Tác vụ phân loại ('multiclass' hoặc 'binary').
        include_test: Có nạp tập test không. Mặc định False.
        max_train_rows: Giới hạn số dòng train (lấy mẫu phân tầng nếu có).
        seed: Seed sinh số ngẫu nhiên.

    Returns:
        ExperimentData: Dữ liệu thực nghiệm đã chuẩn hóa.
    """
    # Lấy X, y gốc cho train
    X_train_raw, y_train_full = to_xy(df, split.train, task=task)
    idx_train = split.train.copy()

    # Lấy mẫu con nếu cần
    if max_train_rows is not None and max_train_rows < len(idx_train):
        sub_idx = subsample_stratified(y_train_full, n=max_train_rows, seed=seed)
        X_train_raw = X_train_raw[sub_idx]
        y_train_full = y_train_full[sub_idx]
        idx_train = idx_train[sub_idx]

    # Fit scaler CHỈ trên train
    scaler = StandardScalerNP(feature_names=FEATURES)
    X_train = scaler.fit_transform(X_train_raw)

    # Trung bình đặc trưng gốc trên train (dùng cho what-if)
    pop_mean_raw = X_train_raw.mean(axis=0).astype(np.float32)

    # Val: chỉ transform
    X_val_raw, y_val = to_xy(df, split.val, task=task)
    X_val = scaler.transform(X_val_raw)

    # Test: chỉ transform nếu cờ include_test được bật
    if include_test:
        X_test_raw, y_test = to_xy(df, split.test, task=task)
        X_test: np.ndarray | None = scaler.transform(X_test_raw)
        idx_test = split.test.copy()
    else:
        X_test = None
        y_test = None
        idx_test = split.test.copy()

    return ExperimentData(
        X_train=X_train,
        y_train=y_train_full,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        X_train_raw=X_train_raw,
        pop_mean_raw=pop_mean_raw,
        idx_train=idx_train,
        idx_val=split.val.copy(),
        idx_test=idx_test,
        task=task,
        variant=split.variant,
        class_names=class_names(task),
    )


def load_experiment_data(
    task: str | None = None,
    variant: str | None = None,
    include_test: bool = False,
    max_train_rows: int | None = None,
) -> ExperimentData:
    """Nạp dữ liệu thực nghiệm từ cấu hình và file lưu trữ.

    Đọc configs/data.yaml, load_raw, load_split(variant), rồi gọi
    build_experiment_data_from_frame.

    Khi bật include_test sẽ ghi warning.

    Args:
        task: Tác vụ ('multiclass' hoặc 'binary'). None → đọc từ config.
        variant: Biến thể ('full' hoặc 'dedup'). None → đọc từ config.
        include_test: Có nạp tập test không (mặc định False).
        max_train_rows: Giới hạn số dòng train.

    Returns:
        ExperimentData: Dữ liệu thực nghiệm.
    """
    config: dict[str, Any] = load_config("data")

    if task is None:
        task = config.get("task", "multiclass")
    if variant is None:
        variant = "dedup" if config.get("dedup", False) else "full"

    seed: int = config.get("split", {}).get("seed", 42)

    if include_test:
        logger.warning("ĐANG NẠP TẬP TEST")

    df = load_raw(config=config)
    split = load_split(variant=variant, seed=seed)

    return build_experiment_data_from_frame(
        df=df,
        split=split,
        task=task,
        include_test=include_test,
        max_train_rows=max_train_rows,
        seed=seed,
    )
