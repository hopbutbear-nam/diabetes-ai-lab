"""Tests cho evaluation/thresholds.py — K05c."""

import numpy as np
import pytest

from evaluation.thresholds import (
    apply_scale,
    select_binary_threshold,
    select_scale_factors,
)


# ---------------------------------------------------------------------------
# Dữ liệu tổng hợp dùng chung
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# 3-class: 300 mẫu, 3 lớp không cân bằng
N = 300
_y3 = RNG.choice([0, 1, 2], size=N, p=[0.5, 0.3, 0.2])
_p3_raw = RNG.dirichlet(alpha=[3, 1, 1], size=N)  # lớp 1,2 rất nhỏ → scale cần thiết
_proba3 = _p3_raw / _p3_raw.sum(axis=1, keepdims=True)

# 2-class: 200 mẫu, mất cân bằng (20 % dương)
N2 = 200
_y2 = (RNG.random(N2) < 0.2).astype(int)
_p2 = RNG.beta(a=2, b=5, size=N2)
_p2[_y2 == 1] += 0.3  # làm cho lớp 1 có xác suất cao hơn một chút
_p2 = np.clip(_p2, 0, 1)


# ---------------------------------------------------------------------------
# Test 1: scale grid không bao giờ kém argmax
# ---------------------------------------------------------------------------

def test_scale_grid_never_worse_than_argmax():
    """Kết quả select_scale_factors phải ≥ baseline argmax trên validation."""
    result = select_scale_factors(_y3, _proba3)
    assert result["val_macro_f1_scaled"] >= result["val_macro_f1_argmax"] - 1e-9, (
        f"Scaled F1 ({result['val_macro_f1_scaled']:.4f}) < "
        f"argmax F1 ({result['val_macro_f1_argmax']:.4f})"
    )
    # Kiểm tra cấu trúc trả về
    for key in ("scale", "val_macro_f1_scaled", "val_macro_f1_argmax", "val_recall", "val_precision"):
        assert key in result, f"Thiếu khoá '{key}' trong kết quả"


# ---------------------------------------------------------------------------
# Test 2: apply_scale thay đổi nhãn khi scale thay đổi
# ---------------------------------------------------------------------------

def test_apply_scale_changes_labels():
    """apply_scale với scale=(1,10,1) phải đẩy nhiều mẫu sang lớp 1."""
    labels_default = apply_scale(_proba3, [1.0, 1.0, 1.0])
    labels_scaled = apply_scale(_proba3, [1.0, 10.0, 1.0])

    # Với scale 10× cho lớp 1, số mẫu dự đoán lớp 1 phải nhiều hơn
    n_class1_default = np.sum(labels_default == 1)
    n_class1_scaled = np.sum(labels_scaled == 1)
    assert n_class1_scaled > n_class1_default, (
        f"scale=(1,10,1) không tăng số mẫu lớp 1: "
        f"{n_class1_default} → {n_class1_scaled}"
    )

    # apply_scale([1,1,1]) phải bằng argmax thuần
    np.testing.assert_array_equal(
        labels_default,
        np.argmax(_proba3, axis=1),
        err_msg="apply_scale([1,1,1]) phải bằng argmax thuần",
    )


# ---------------------------------------------------------------------------
# Test 3: binary threshold tối đa hóa F1
# ---------------------------------------------------------------------------

def test_binary_threshold_maximizes_f1():
    """Ngưỡng tìm được phải cho macro-F1 ≥ ngưỡng 0.5 (baseline)."""
    result = select_binary_threshold(_y2, _p2, objective="macro_f1")

    # Kiểm tra cấu trúc
    for key in ("threshold", "val_macro_f1", "val_recall", "val_precision", "objective"):
        assert key in result

    assert 0.0 < result["threshold"] < 1.0
    assert result["objective"] == "macro_f1"

    # So sánh với threshold = 0.5
    from sklearn.metrics import f1_score
    f1_at_05 = f1_score(_y2, (_p2 >= 0.5).astype(int), average="macro", zero_division=0)
    assert result["val_macro_f1"] >= f1_at_05 - 1e-9, (
        f"Macro-F1 tối ưu ({result['val_macro_f1']:.4f}) < F1 tại 0.5 ({f1_at_05:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: recall_at_precision tôn trọng precision_floor
# ---------------------------------------------------------------------------

def test_recall_at_precision_respects_floor():
    """recall_at_precision phải cho precision ≥ floor; ValueError nếu không đạt."""
    floor = 0.30
    result = select_binary_threshold(
        _y2, _p2, objective="recall_at_precision", precision_floor=floor
    )
    assert result["val_precision"] >= floor - 1e-9, (
        f"Precision ({result['val_precision']:.4f}) < floor ({floor})"
    )
    assert result["objective"] == "recall_at_precision"

    # Với floor không thể đạt (> 1.0) → phải raise ValueError
    with pytest.raises(ValueError, match="precision"):
        select_binary_threshold(
            _y2, _p2, objective="recall_at_precision", precision_floor=1.1
        )
