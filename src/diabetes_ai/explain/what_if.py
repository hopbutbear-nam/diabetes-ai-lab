"""Mô đun phân tích tác động đặc trưng và giả lập what-if lối sống.

Lưu ý: Thay đổi theo mô hình học từ khảo sát (tương quan), không phải bằng chứng nhân quả.
"""

from typing import Any, Callable
import numpy as np

from diabetes_ai.data.schema import FEATURE_RANGES, FEATURES
from diabetes_ai.features.feature_names import vi

WHAT_IF_NOTE: str = (
    "Đây là thay đổi theo mô hình học từ khảo sát (tương quan), không phải bằng chứng nhân quả hay lời hứa về hiệu quả."
)


def _eval_index(index_fn: Callable[[np.ndarray], np.ndarray], vec: np.ndarray) -> float:
    """Đánh giá chỉ số nguy cơ cho một vector mẫu."""
    mat = np.atleast_2d(vec)
    res = index_fn(mat)
    arr = np.asarray(res)
    return float(arr.flatten()[0])


def feature_effects(
    index_fn: Callable[[np.ndarray], np.ndarray],
    x_raw: np.ndarray,
    pop_mean_raw: np.ndarray,
    features: list[str] | None = None,
    skip: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Tính mức đóng góp của từng đặc trưng so với trung bình quần thể.

    Args:
        index_fn: Hàm tính chỉ số nguy cơ nhận ma trận đặc trưng chưa chuẩn hóa.
        x_raw: Vector đặc trưng thô của cá nhân.
        pop_mean_raw: Vector giá trị trung bình quần thể thô.
        features: Danh sách tên đặc trưng theo thứ tự vector. Mặc định là FEATURES.
        skip: Tập hợp đặc trưng bỏ qua (ví dụ drop_for_what_if).

    Returns:
        list[dict[str, Any]]: Danh sách tác động đặc trưng sắp theo |effect| giảm dần.
    """
    if features is None:
        features = list(FEATURES)
    if skip is None:
        skip = set()

    x = np.asarray(x_raw, dtype=float).flatten()
    pop_mean = np.asarray(pop_mean_raw, dtype=float).flatten()

    base_idx = _eval_index(index_fn, x)

    results: list[dict[str, Any]] = []
    for j, feat in enumerate(features):
        if feat in skip:
            continue
        x2 = x.copy()
        x2[j] = pop_mean[j]
        idx2 = _eval_index(index_fn, x2)
        effect = float(base_idx - idx2)
        results.append({
            "feature": feat,
            "vi": vi(feat),
            "value": float(x[j]),
            "effect": effect,
        })

    results.sort(key=lambda item: abs(item["effect"]), reverse=True)
    return results


def what_if_scenarios(
    index_fn: Callable[[np.ndarray], np.ndarray],
    x_raw: np.ndarray,
    cfg: dict[str, Any],
    features: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Tạo các kịch bản what-if lối sống dựa trên cấu hình.

    Bỏ qua nếu cá nhân đã đạt ngưỡng lành mạnh (ví dụ BMI <= target + min_gap, hoặc đã không hút thuốc).

    Args:
        index_fn: Hàm tính chỉ số nguy cơ nhận ma trận đặc trưng chưa chuẩn hóa.
        x_raw: Vector đặc trưng thô của cá nhân.
        cfg: Cấu hình what_if (hoặc dict chứa mục 'what_if').
        features: Danh sách tên đặc trưng theo thứ tự vector. Mặc định là FEATURES.

    Returns:
        list[dict[str, Any]]: Danh sách các kịch bản cải thiện lối sống, sắp theo delta_index giảm dần.
    """
    if features is None:
        features = list(FEATURES)

    what_if_cfg = cfg.get("what_if", cfg) if isinstance(cfg, dict) else cfg
    x = np.asarray(x_raw, dtype=float).flatten()
    feat_to_idx = {name: i for i, name in enumerate(features)}

    base_idx = _eval_index(index_fn, x)

    scenarios: list[dict[str, Any]] = []
    for feat, spec in what_if_cfg.items():
        if feat not in feat_to_idx:
            continue
        idx = feat_to_idx[feat]
        cur_val = float(x[idx])
        target_val = float(spec["target"])
        min_gap = float(spec.get("min_gap", 0.0))
        action = str(spec.get("action", f"Đưa {feat} về {target_val}"))

        # Bỏ qua nếu đã ở mức lành mạnh
        if spec.get("kind") == "continuous" or "min_gap" in spec:
            if cur_val <= target_val + min_gap:
                continue
        else:
            if target_val == 0.0 and cur_val <= 0.0:
                continue
            if target_val == 1.0 and cur_val >= 1.0:
                continue
            if cur_val == target_val:
                continue

        x_target = x.copy()
        x_target[idx] = target_val
        target_idx = _eval_index(index_fn, x_target)
        delta_index = float(base_idx - target_idx)

        scenarios.append({
            "feature": feat,
            "action": action,
            "from_value": cur_val,
            "to_value": target_val,
            "delta_index": delta_index,
            "note": WHAT_IF_NOTE,
        })

    scenarios.sort(key=lambda item: item["delta_index"], reverse=True)
    return scenarios


def what_if_custom(
    index_fn: Callable[[np.ndarray], np.ndarray],
    x_raw: np.ndarray,
    changes: dict[str, float],
    allowed: set[str],
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Giả lập thay đổi tùy biến trên các đặc trưng được phép.

    Args:
        index_fn: Hàm tính chỉ số nguy cơ nhận ma trận đặc trưng chưa chuẩn hóa.
        x_raw: Vector đặc trưng thô ban đầu.
        changes: Từ điển các đặc trưng và giá trị mới muốn thay đổi.
        allowed: Tập hợp các tên đặc trưng được phép thay đổi.
        features: Danh sách tên đặc trưng theo thứ tự vector. Mặc định là FEATURES.

    Returns:
        dict[str, Any]: Kết quả gồm 'before', 'after', 'delta' và 'note'.

    Raises:
        ValueError: Nếu có đặc trưng nằm ngoài danh sách allowed hoặc giá trị nằm ngoài FEATURE_RANGES.
    """
    if features is None:
        features = list(FEATURES)

    invalid_feats = [k for k in changes if k not in allowed]
    if invalid_feats:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(
            f"Chỉ được thay đổi: {allowed_str}. Đặc trưng không được phép: {sorted(invalid_feats)}"
        )

    for feat, val in changes.items():
        if feat in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feat]
            if val < min_val or val > max_val:
                raise ValueError(
                    f"Giá trị của {feat} ({val}) nằm ngoài khoảng cho phép [{min_val}, {max_val}]."
                )

    x = np.asarray(x_raw, dtype=float).flatten()
    feat_to_idx = {name: i for i, name in enumerate(features)}

    before = _eval_index(index_fn, x)

    x_new = x.copy()
    for feat, val in changes.items():
        if feat not in feat_to_idx:
            raise ValueError(f"Đặc trưng {feat} không tồn tại trong danh sách đặc trưng.")
        x_new[feat_to_idx[feat]] = float(val)

    after = _eval_index(index_fn, x_new)
    delta = float(before - after)

    return {
        "before": before,
        "after": after,
        "delta": delta,
        "note": WHAT_IF_NOTE,
    }
