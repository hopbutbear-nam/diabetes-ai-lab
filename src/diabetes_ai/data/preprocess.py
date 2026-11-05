"""Tiền xử lý dữ liệu: mã hóa nhãn mục tiêu và xử lý dòng trùng lặp."""

import numpy as np
import pandas as pd


def encode_target(
    y: np.ndarray | pd.Series,
    task: str = "multiclass",
) -> np.ndarray:
    """Mã hóa nhãn mục tiêu cho bài toán phân loại.

    Args:
        y: Mảng hoặc Series nhãn ban đầu (0: không, 1: tiền tiểu đường, 2: tiểu đường).
        task: Tác vụ phân loại ('multiclass' hoặc 'binary').

    Returns:
        np.ndarray: Mảng nhãn kiểu int64 đã được mã hóa.

    Raises:
        ValueError: Nếu task không phải 'multiclass' hoặc 'binary'.
    """
    arr = np.asarray(y)
    if task == "multiclass":
        return arr.astype(np.int64)
    elif task == "binary":
        return (arr > 0).astype(np.int64)
    else:
        raise ValueError(
            f"task phải là 'multiclass' hoặc 'binary', nhận được: '{task}'."
        )


def duplicate_mask(df: pd.DataFrame) -> np.ndarray:
    """Tạo mặt nạ boolean đánh dấu các dòng trùng lặp hoàn toàn trên toàn bộ cột.

    Đánh dấu True cho bản sao thứ hai trở đi (keep='first').

    Args:
        df: DataFrame cần kiểm tra.

    Returns:
        np.ndarray: Mảng boolean 1D với True là các dòng trùng lặp.
    """
    return df.duplicated(keep="first").to_numpy()


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các dòng trùng lặp hoàn toàn và giữ nguyên chỉ số (index) gốc.

    Args:
        df: DataFrame ban đầu.

    Returns:
        pd.DataFrame: DataFrame sau khi loại bỏ dòng trùng (giữ bản ghi đầu tiên, không reset index).
    """
    return df.drop_duplicates(keep="first")
