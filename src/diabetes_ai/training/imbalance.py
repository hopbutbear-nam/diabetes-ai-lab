"""Xử lý mất cân bằng lớp: trọng số lớp, trọng số mẫu và lấy mẫu con phân tầng."""

import numpy as np


def class_weights(
    y: np.ndarray, n_classes: int | None = None, power: float = 0.75
) -> np.ndarray:
    """Tính trọng số lớp theo công thức nghịch đảo lũy thừa tần suất.

    Args:
        y: Mảng nhãn 1D.
        n_classes: Số lượng lớp (nếu None sẽ lấy max(y) + 1).
        power: Lũy thừa làm mượt trọng số (mặc định 0.75).

    Returns:
        np.ndarray: Trọng số cho từng lớp.

    Raises:
        ValueError: Nếu có lớp không có mẫu trong tập huấn luyện.
    """
    y = np.asarray(y)
    k = n_classes or int(y.max()) + 1
    counts = np.bincount(y, minlength=k).astype("float64")
    if (counts == 0).any():
        raise ValueError(
            f"Lớp {int(np.argmin(counts))} không có mẫu trong tập huấn luyện"
        )
    return (len(y) / (k * counts)) ** power


def sample_weights(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gán trọng số mẫu tương ứng với nhãn từng mẫu.

    Args:
        y: Mảng nhãn 1D.
        weights: Trọng số của từng lớp.

    Returns:
        np.ndarray: Trọng số của từng mẫu (weights[y]).
    """
    y = np.asarray(y)
    weights = np.asarray(weights)
    return weights[y]


def subsample_stratified(y: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Lấy mẫu con phân tầng giữ tỉ lệ lớp và mỗi lớp có ít nhất 1 mẫu.

    Args:
        y: Mảng nhãn 1D.
        n: Số lượng mẫu con cần lấy.
        seed: Seed sinh số ngẫu nhiên (mặc định 42).

    Returns:
        np.ndarray: Mảng chỉ số đã sắp xếp tăng dần.

    Raises:
        ValueError: Nếu n nhỏ hơn số lớp.
    """
    y = np.asarray(y)
    total_len = len(y)
    if n >= total_len:
        return np.arange(total_len)

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n < n_classes:
        raise ValueError(
            f"Số mẫu cần lấy n={n} nhỏ hơn số lớp {n_classes} (mỗi lớp cần ít nhất 1 mẫu)"
        )

    rng = np.random.default_rng(seed)

    # Phân bổ số mẫu cho từng lớp theo tỉ lệ
    proportions = counts / total_len
    raw_targets = n * proportions
    allocated = np.floor(raw_targets).astype(int)
    allocated = np.maximum(allocated, 1)
    allocated = np.minimum(allocated, counts)

    diff = n - int(allocated.sum())

    while diff > 0:
        can_add = np.where(allocated < counts)[0]
        if len(can_add) == 0:
            break
        remainders = raw_targets - allocated
        sorted_indices = can_add[np.argsort(-remainders[can_add])]
        for idx in sorted_indices:
            allocated[idx] += 1
            diff -= 1
            if diff == 0:
                break

    while diff < 0:
        can_reduce = np.where(allocated > 1)[0]
        if len(can_reduce) == 0:
            break
        remainders = allocated - raw_targets
        sorted_indices = can_reduce[np.argsort(remainders[can_reduce])]
        for idx in sorted_indices:
            allocated[idx] -= 1
            diff += 1
            if diff == 0:
                break

    selected = []
    for c, count in zip(classes, allocated):
        idx_c = np.where(y == c)[0]
        chosen = rng.choice(idx_c, size=count, replace=False)
        selected.append(chosen)

    all_selected = np.concatenate(selected)
    return np.sort(all_selected)
