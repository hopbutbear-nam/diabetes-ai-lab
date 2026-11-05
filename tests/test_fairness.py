"""Kiểm thử tính công bằng theo nhóm và cấu hình đánh giá (K06a)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from diabetes_ai.evaluation.confidence_intervals import wilson_interval
from diabetes_ai.evaluation.fairness import (
    disparity,
    fairness_report,
    group_codes,
    group_metrics,
)


def test_group_mappings_cover_all_codes():
    """Kiểm tra mã tuổi (1–13), thu nhập (1–8), giới (0–1) mỗi mã thuộc đúng một nhóm."""
    cfg_path = Path("configs/evaluate.yaml")
    assert cfg_path.exists(), "configs/evaluate.yaml không tồn tại."

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fairness_cfg = cfg.get("fairness", {})

    # 1. Tuổi (Age): 1 đến 13
    age_groups = fairness_cfg.get("age_groups", {})
    all_age_codes: list[int] = []
    for g, codes in age_groups.items():
        all_age_codes.extend(codes)
    assert sorted(all_age_codes) == list(range(1, 14)), "Mã tuổi phải bao phủ đủ 1-13 không trùng lặp."
    assert len(all_age_codes) == len(set(all_age_codes)), "Có mã tuổi bị trùng lặp giữa các nhóm."

    # 2. Thu nhập (Income): 1 đến 8
    income_groups = fairness_cfg.get("income_groups", {})
    all_income_codes: list[int] = []
    for g, codes in income_groups.items():
        all_income_codes.extend(codes)
    assert sorted(all_income_codes) == list(range(1, 9)), "Mã thu nhập phải bao phủ đủ 1-8 không trùng lặp."
    assert len(all_income_codes) == len(set(all_income_codes)), "Có mã thu nhập bị trùng lặp giữa các nhóm."

    # 3. Giới tính (Sex): 0 và 1
    sex_groups = fairness_cfg.get("sex_groups", {})
    all_sex_codes: list[int] = []
    for g, codes in sex_groups.items():
        all_sex_codes.extend(codes)
    assert sorted(all_sex_codes) == [0, 1], "Mã giới tính phải bao gồm chính xác 0 và 1."
    assert len(all_sex_codes) == len(set(all_sex_codes)), "Mã giới tính bị trùng lặp."

    # 4. Kiểm tra group_codes ánh xạ thành công và ném lỗi khi gặp mã lạ
    df = pd.DataFrame({"Age": list(range(1, 14))})
    labels = group_codes(df, "Age", age_groups)
    assert len(labels) == 13
    assert labels[0] == "18-34"
    assert labels[-1] == "75+"

    df_invalid = pd.DataFrame({"Age": [99]})
    with pytest.raises(ValueError, match="99"):
        group_codes(df_invalid, "Age", age_groups)


def test_group_metrics_wilson_and_flags():
    """Kiểm tra tính toán chỉ số theo nhóm, khoảng Wilson và cờ cảnh báo."""
    # Tạo nhóm G1 gồm 100 mẫu
    # 40 mẫu dương (y_true = 2), 60 mẫu âm (y_true = 0)
    # y_pred dự đoán 30 mẫu dương: 20 TP (y_true=2), 10 FP (y_true=0)
    # risk_flag gắn cờ 35 mẫu
    y_true = np.array([2] * 40 + [0] * 60)
    y_pred = np.array([2] * 20 + [0] * 20 + [2] * 10 + [0] * 50)
    risk_flag = np.array([True] * 35 + [False] * 65)
    groups = np.array(["G1"] * 100)

    rows = group_metrics(
        groups=groups,
        y_true=y_true,
        y_pred=y_pred,
        risk_flag=risk_flag,
        positive_class=2,
        min_pos=30,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["group"] == "G1"
    assert row["n"] == 100
    assert row["n_pos"] == 40
    assert row["small_sample"] is False

    # Prevalence = 40 / 100 = 0.40
    assert row["prevalence"]["point"] == pytest.approx(0.40)
    w_prev_low, w_prev_high = wilson_interval(40, 100)
    assert row["prevalence"]["low"] == pytest.approx(w_prev_low)
    assert row["prevalence"]["high"] == pytest.approx(w_prev_high)
    assert row["prevalence"]["low"] <= row["prevalence"]["point"] <= row["prevalence"]["high"]

    # Recall = 20 / 40 = 0.50
    assert row["recall"]["point"] == pytest.approx(0.50)
    w_rec_low, w_rec_high = wilson_interval(20, 40)
    assert row["recall"]["low"] == pytest.approx(w_rec_low)
    assert row["recall"]["high"] == pytest.approx(w_rec_high)

    # Precision = 20 / 30 = 0.6667
    assert row["precision"]["point"] == pytest.approx(20 / 30)
    w_prec_low, w_prec_high = wilson_interval(20, 30)
    assert row["precision"]["low"] == pytest.approx(w_prec_low)
    assert row["precision"]["high"] == pytest.approx(w_prec_high)

    # Alert rate = 35 / 100 = 0.35
    assert row["alert_rate"]["point"] == pytest.approx(0.35)
    w_al_low, w_al_high = wilson_interval(35, 100)
    assert row["alert_rate"]["low"] == pytest.approx(w_al_low)
    assert row["alert_rate"]["high"] == pytest.approx(w_al_high)


def test_small_group_is_flagged():
    """Kiểm tra nhóm có n_pos < min_pos được gắn cờ small_sample=True."""
    # Nhóm SmallGroup có 10 mẫu dương (< 30)
    # Nhóm LargeGroup có 35 mẫu dương (>= 30)
    groups = np.array(["SmallGroup"] * 50 + ["LargeGroup"] * 100)
    y_true = np.array([2] * 10 + [0] * 40 + [2] * 35 + [0] * 65)

    rows = group_metrics(groups=groups, y_true=y_true, positive_class=2, min_pos=30)
    assert len(rows) == 2

    by_group = {r["group"]: r for r in rows}
    assert by_group["SmallGroup"]["n_pos"] == 10
    assert by_group["SmallGroup"]["small_sample"] is True

    assert by_group["LargeGroup"]["n_pos"] == 35
    assert by_group["LargeGroup"]["small_sample"] is False


def test_disparity_ignores_small_groups():
    """Kiểm tra tính chênh lệch disparity bỏ qua nhóm có small_sample=True."""
    rows = [
        {
            "group": "GroupA",
            "n": 200,
            "n_pos": 50,
            "small_sample": False,
            "recall": {"point": 0.85, "low": 0.75, "high": 0.92},
        },
        {
            "group": "GroupB",
            "n": 200,
            "n_pos": 45,
            "small_sample": False,
            "recall": {"point": 0.70, "low": 0.58, "high": 0.80},
        },
        {
            "group": "GroupC",
            "n": 50,
            "n_pos": 5,
            "small_sample": True,
            "recall": {"point": 0.10, "low": 0.02, "high": 0.30},
        },
    ]

    disp = disparity(rows, metric="recall")

    assert disp["ignored_groups"] == ["GroupC"]
    assert disp["best_group"] == "GroupA"
    assert disp["worst_group"] == "GroupB"
    # max_gap_pp: (0.85 - 0.70) * 100 = 15.0 percentage points
    assert disp["max_gap_pp"] == pytest.approx(15.0)

    # Kiểm tra thêm khi chạy qua fairness_report
    df_feat = pd.DataFrame({
        "Age": [1] * 200 + [4] * 200 + [12] * 50,
    })
    y_true = np.array([2] * 50 + [0] * 150 + [2] * 45 + [0] * 155 + [2] * 5 + [0] * 45)
    y_pred = y_true.copy()  # perfect predictions
    rep = fairness_report(df_feat, y_true, y_pred, min_pos=30)
    assert "Age" in rep
    assert rep["Age"]["by_class"]["diabetes"]["disparity"]["ignored_groups"] == ["75+"]
