"""9 test cho evaluation/metrics.py."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from diabetes_ai.evaluation.metrics import (
    brier_multiclass,
    compute_all,
    ece_binary,
    ece_ovr,
    macro_f1,
    majority_baseline_accuracy,
    pr_auc_ovr,
    recall_precision_f1_per_class,
    roc_auc_ovr,
    roc_auc_per_class,
)


# ---------------------------------------------------------------------------
# test_auc_perfect_and_random
# ---------------------------------------------------------------------------

def test_auc_perfect_and_random() -> None:
    """AUC hoàn hảo = 1.0; AUC ngẫu nhiên nằm trong 0.5 ± 0.03."""
    rng = np.random.default_rng(42)
    n = 1000
    K = 3

    # Hoàn hảo: proba[i, y_i] = 1
    y_perf = rng.integers(0, K, size=n).astype(np.int64)
    p_perf = np.zeros((n, K), dtype=np.float64)
    p_perf[np.arange(n), y_perf] = 1.0
    auc_perf = roc_auc_ovr(y_perf, p_perf)
    assert abs(auc_perf - 1.0) < 1e-9, f"AUC hoàn hảo phải = 1.0, nhận {auc_perf}"

    # Ngẫu nhiên: proba đều
    p_rand = np.ones((n, K), dtype=np.float64) / K
    # Thêm nhiễu nhỏ để có thứ hạng
    p_rand += rng.uniform(0, 0.01, size=(n, K))
    p_rand /= p_rand.sum(axis=1, keepdims=True)
    y_rand = rng.integers(0, K, size=n).astype(np.int64)
    auc_rand = roc_auc_ovr(y_rand, p_rand)
    assert abs(auc_rand - 0.5) <= 0.03, f"AUC ngẫu nhiên phải ~ 0.5 ± 0.03, nhận {auc_rand}"


# ---------------------------------------------------------------------------
# test_auc_binary_matches_sklearn
# ---------------------------------------------------------------------------

def test_auc_binary_matches_sklearn() -> None:
    """Với K=2, roc_auc_ovr phải khớp với sklearn trên cột 1."""
    rng = np.random.default_rng(7)
    n = 500
    y = rng.integers(0, 2, size=n).astype(np.int64)
    p_col1 = rng.uniform(0, 1, size=n)
    p_col0 = 1.0 - p_col1
    proba = np.stack([p_col0, p_col1], axis=1)

    result = roc_auc_ovr(y, proba)
    expected = roc_auc_score(y, p_col1)
    assert abs(result - expected) < 1e-9, f"roc_auc_ovr K=2 không khớp sklearn: {result} vs {expected}"


# ---------------------------------------------------------------------------
# test_macro_f1_known
# ---------------------------------------------------------------------------

def test_macro_f1_known() -> None:
    """Kiểm tra macro_f1 với trường hợp biết trước."""
    # Tất cả đúng → F1 = 1.0
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    assert abs(macro_f1(y, y_pred, 3) - 1.0) < 1e-9

    # Đoán sai hết → recall = 0, precision = 0 → F1 = 0
    y = np.array([0, 0, 0, 0], dtype=np.int64)
    y_pred = np.array([1, 1, 1, 1], dtype=np.int64)
    f1 = macro_f1(y, y_pred, 3)
    assert f1 == pytest.approx(0.0), f"Macro F1 phải = 0 khi đoán sai hết, nhận {f1}"


# ---------------------------------------------------------------------------
# test_recall_precision_per_class_known
# ---------------------------------------------------------------------------

def test_recall_precision_per_class_known() -> None:
    """Kiểm tra recall, precision, f1 từng lớp với ví dụ cụ thể."""
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
    result = recall_precision_f1_per_class(y_true, y_pred, 3)

    # Lớp 0: TP=1, FN=1, FP=0 → recall=0.5, precision=1.0, f1=2/3
    assert result["recall"][0] == pytest.approx(0.5)
    assert result["precision"][0] == pytest.approx(1.0)
    assert result["f1"][0] == pytest.approx(2.0 / 3.0, abs=1e-9)

    # Lớp 1: TP=1, FN=1, FP=1 → recall=0.5, precision=0.5, f1=0.5
    assert result["recall"][1] == pytest.approx(0.5)
    assert result["precision"][1] == pytest.approx(0.5)
    assert result["f1"][1] == pytest.approx(0.5)

    # Lớp 2: TP=2, FN=0, FP=1 → recall=1.0, precision=2/3, f1=4/5
    assert result["recall"][2] == pytest.approx(1.0)
    assert result["precision"][2] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert result["f1"][2] == pytest.approx(4.0 / 5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# test_brier_multiclass_known
# ---------------------------------------------------------------------------

def test_brier_multiclass_known() -> None:
    """Brier hoàn hảo = 0; Brier đều 1/3 với K=3 = 2/3."""
    K = 3
    n = 90

    # Hoàn hảo
    y_perf = np.tile([0, 1, 2], n // 3).astype(np.int64)
    p_perf = np.zeros((len(y_perf), K))
    p_perf[np.arange(len(y_perf)), y_perf] = 1.0
    bs_perf = brier_multiclass(y_perf, p_perf)
    assert bs_perf == pytest.approx(0.0, abs=1e-9), f"Brier hoàn hảo phải = 0, nhận {bs_perf}"

    # Đều 1/3: Σ_k (1/3 − 1[y=k])² = (1/3 - 1)² + 2*(1/3)² = 4/9 + 2/9 = 6/9 = 2/3
    p_uniform = np.full((len(y_perf), K), 1.0 / K)
    bs_uniform = brier_multiclass(y_perf, p_uniform)
    assert bs_uniform == pytest.approx(2.0 / 3.0, abs=1e-9), f"Brier đều 1/3 phải = 2/3, nhận {bs_uniform}"


# ---------------------------------------------------------------------------
# test_ece_zero_when_calibrated
# ---------------------------------------------------------------------------

def test_ece_zero_when_calibrated() -> None:
    """ECE ≈ 0 khi mô hình được hiệu chuẩn hoàn hảo (p = tỉ lệ dương thực tế)."""
    n = 1000
    # Bin 1: p=0.3, 30% dương
    p = np.array([0.3] * n, dtype=np.float64)
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=n) < 0.3).astype(np.float64)
    ece = ece_binary(y, p, n_bins=10)
    # Với n lớn, ECE gần 0; cho phép sai số nhỏ do ngẫu nhiên
    assert ece < 0.05, f"ECE khi hiệu chuẩn tốt phải nhỏ, nhận {ece}"


# ---------------------------------------------------------------------------
# test_ece_large_when_overconfident
# ---------------------------------------------------------------------------

def test_ece_large_when_overconfident() -> None:
    """ECE ≈ 0.4 khi mô hình tự tin quá (p=0.9 nhưng ~50% dương)."""
    n = 1000
    rng = np.random.default_rng(1)
    p = np.full(n, 0.9, dtype=np.float64)
    y = (rng.uniform(size=n) < 0.5).astype(np.float64)
    ece = ece_binary(y, p, n_bins=10)
    assert ece >= 0.35, f"ECE overconfident phải ≥ 0.35, nhận {ece}"


# ---------------------------------------------------------------------------
# test_majority_baseline_accuracy
# ---------------------------------------------------------------------------

def test_majority_baseline_accuracy() -> None:
    """Kiểm tra majority_baseline_accuracy với ví dụ biết trước."""
    # 70 lớp 0, 20 lớp 1, 10 lớp 2 → baseline = 0.7
    y = np.array([0] * 70 + [1] * 20 + [2] * 10, dtype=np.int64)
    result = majority_baseline_accuracy(y)
    assert result == pytest.approx(0.7, abs=1e-9)

    # Tất cả cùng lớp → baseline = 1.0
    y_all = np.zeros(50, dtype=np.int64)
    assert majority_baseline_accuracy(y_all) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_compute_all_keys_and_baseline
# ---------------------------------------------------------------------------

def test_compute_all_keys_and_baseline() -> None:
    """Kiểm tra compute_all trả đủ khóa và accuracy_majority_baseline đúng."""
    rng = np.random.default_rng(99)
    n, K = 200, 3
    y = rng.integers(0, K, size=n).astype(np.int64)
    proba = rng.dirichlet(np.ones(K), size=n).astype(np.float64)

    result = compute_all(y, proba)

    expected_keys = {
        "n", "accuracy", "accuracy_majority_baseline",
        "roc_auc_ovr_macro", "roc_auc_per_class",
        "pr_auc_macro", "pr_auc_per_class",
        "macro_f1", "recall_per_class", "precision_per_class", "f1_per_class",
        "brier_multiclass", "brier_per_class",
        "ece_per_class", "ece_macro",
        "confusion_matrix",
    }
    assert expected_keys.issubset(result.keys()), f"Thiếu khóa: {expected_keys - result.keys()}"
    assert result["n"] == n

    # Baseline phải ≥ accuracy của mô hình ngẫu nhiên hoặc ít nhất tính đúng
    baseline = result["accuracy_majority_baseline"]
    counts = np.bincount(y, minlength=K)
    expected_baseline = float(counts.max() / n)
    assert abs(baseline - expected_baseline) < 1e-9
