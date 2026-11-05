"""Kiểm thử biểu đồ đánh giá và báo cáo số liệu."""

from pathlib import Path

import numpy as np
import pytest

from diabetes_ai.evaluation.plots import (
    plot_confusion,
    plot_fairness_bars,
    plot_learning_curve,
    plot_pr_ovr,
    plot_reliability,
    plot_roc_ovr,
)
from diabetes_ai.evaluation.report import (
    acquire_test_lock,
    file_sha256,
    read_lock,
    risk_alert_stats,
    write_json,
)


def test_plots_write_png(tmp_path: Path) -> None:
    """Kiểm tra các hàm vẽ biểu đồ tạo file PNG hợp lệ (> 1 KB)."""
    np.random.seed(42)
    n = 100
    y = np.random.choice([0, 1, 2], size=n, p=[0.7, 0.15, 0.15])
    proba = np.random.dirichlet(np.ones(3), size=n)
    class_names = ["Không", "Tiền ĐTĐ", "ĐTĐ"]

    # 1. plot_roc_ovr
    roc_path = plot_roc_ovr(y, proba, class_names, tmp_path / "roc.png")
    assert roc_path.is_file()
    assert roc_path.stat().st_size > 1000

    # 2. plot_confusion
    cm = np.array([[60, 5, 5], [4, 10, 1], [3, 2, 10]])
    cm_path = plot_confusion(cm, class_names, tmp_path / "cm.png")
    assert cm_path.is_file()
    assert cm_path.stat().st_size > 1000

    # 3. plot_pr_ovr
    pr_path = plot_pr_ovr(y, proba, class_names, tmp_path / "pr.png")
    assert pr_path.is_file()
    assert pr_path.stat().st_size > 1000

    # 4. plot_reliability
    rel_path = plot_reliability(y, proba, class_names, tmp_path / "rel.png")
    assert rel_path.is_file()
    assert rel_path.stat().st_size > 1000

    # 5. plot_learning_curve
    history = [
        {"epoch": 1, "train_loss": 0.50, "val_loss": 0.48, "val_auc_ovr": 0.70},
        {"epoch": 2, "train_loss": 0.45, "val_loss": 0.42, "val_auc_ovr": 0.75},
        {"epoch": 3, "train_loss": 0.40, "val_loss": 0.43, "val_auc_ovr": 0.74},
    ]
    lc_path = plot_learning_curve(history, tmp_path / "lc.png")
    assert lc_path.is_file()
    assert lc_path.stat().st_size > 1000

    # 6. plot_fairness_bars
    rows_by_group = {
        "Sex": [
            {
                "group": "Nữ",
                "recall": {"point": 0.60, "low": 0.50, "high": 0.70},
                "small_sample": False,
            },
            {
                "group": "Nam",
                "recall": {"point": 0.65, "low": 0.55, "high": 0.75},
                "small_sample": False,
            },
        ]
    }
    fair_path = plot_fairness_bars(rows_by_group, "recall", tmp_path / "fair.png")
    assert fair_path.is_file()
    assert fair_path.stat().st_size > 1000

    # Kiểm tra các tiện ích khóa và JSON
    dummy_file = tmp_path / "sample.txt"
    dummy_file.write_text("kiểm thử hash sha256", encoding="utf-8")
    assert len(file_sha256(dummy_file)) == 64

    lock_file = tmp_path / "test_lock.json"
    assert read_lock(lock_file) == {"evaluations": []}

    runs = ["multiclass_full"]
    hashes = {"model.joblib": "dummy_hash"}
    res_lock = acquire_test_lock(lock_file, runs, hashes)
    assert len(res_lock["evaluations"]) == 1
    assert res_lock["evaluations"][0]["n"] == 1

    # Lần thứ hai không force phải báo lỗi
    with pytest.raises(RuntimeError, match="Test đã được đánh giá"):
        acquire_test_lock(lock_file, runs, hashes, force=False)

    # Có force thì ghi nhận lần 2
    res_forced = acquire_test_lock(lock_file, runs, hashes, force=True, reason="chạy lại kiểm thử")
    assert len(res_forced["evaluations"]) == 2
    assert res_forced["evaluations"][1]["forced"] is True
    assert res_forced["evaluations"][1]["reason"] == "chạy lại kiểm thử"


def test_risk_alert_stats_known() -> None:
    """Kiểm tra risk_alert_stats với dữ liệu biết trước cho alert_rate, ppv, lift."""
    # 6 mẫu: 4 âm (lớp 0), 2 dương (lớp 2) -> tỉ lệ dương nền = 2/6 = 1/3
    y_true = np.array([0, 0, 0, 0, 2, 2])
    risk_index = np.array([0.05, 0.08, 0.15, 0.20, 0.30, 0.40])

    thresholds = {
        "medium_or_above": 0.10,
        "high": 0.25,
    }
    stats = risk_alert_stats(y_true, risk_index, thresholds, positive_class=2)

    assert stats["n"] == 6
    assert stats["n_pos"] == 2
    assert pytest.approx(stats["base_rate"], abs=1e-5) == 2 / 6

    # Với ngưỡng 0.10: các mẫu [0.15, 0.20, 0.30, 0.40] được cảnh báo (4 mẫu)
    # Trong đó có 2 mẫu dương thực tế -> TP=2, FP=2
    # alert_rate = 4/6 = 2/3
    # ppv = 2/4 = 0.5
    # recall = 2/2 = 1.0
    # FPR = 2/4 = 0.5
    # lift = 0.5 / (1/3) = 1.5
    med = stats["medium_or_above"]
    assert pytest.approx(med["alert_rate"], abs=1e-5) == 4 / 6
    assert pytest.approx(med["ppv"], abs=1e-5) == 2 / 4
    assert pytest.approx(med["recall"], abs=1e-5) == 1.0
    assert pytest.approx(med["false_positive_rate"], abs=1e-5) == 2 / 4
    assert pytest.approx(med["lift"], abs=1e-5) == 1.5

    # Với ngưỡng 0.25: các mẫu [0.30, 0.40] được cảnh báo (2 mẫu)
    # Cả 2 đều dương thực tế -> TP=2, FP=0
    # alert_rate = 2/6 = 1/3
    # ppv = 2/2 = 1.0
    # recall = 2/2 = 1.0
    # FPR = 0/4 = 0.0
    # lift = 1.0 / (1/3) = 3.0
    high = stats["high"]
    assert pytest.approx(high["alert_rate"], abs=1e-5) == 2 / 6
    assert pytest.approx(high["ppv"], abs=1e-5) == 1.0
    assert pytest.approx(high["recall"], abs=1e-5) == 1.0
    assert pytest.approx(high["false_positive_rate"], abs=1e-5) == 0.0
    assert pytest.approx(high["lift"], abs=1e-5) == 3.0
