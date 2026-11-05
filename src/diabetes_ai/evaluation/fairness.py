"""Đánh giá tính công bằng (fairness) theo các nhóm nhân khẩu học kèm khoảng tin cậy Wilson."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from diabetes_ai.evaluation.confidence_intervals import wilson_interval


def group_codes(
    df_features: pd.DataFrame | pd.Series | dict[str, Any],
    column: str,
    mapping: dict[str, list[int]],
) -> np.ndarray:
    """Ánh xạ các mã số đặc trưng sang tên nhóm tương ứng.

    Args:
        df_features: DataFrame hoặc đối tượng chứa cột cần nhóm.
        column: Tên cột cần phân nhóm (ví dụ: 'Age', 'Income', 'Sex').
        mapping: Ánh xạ từ tên nhóm sang danh sách các mã (ví dụ: {"18-34": [1, 2, 3]}).

    Returns:
        np.ndarray: Mảng nhãn nhóm (dạng chuỗi) cho từng hàng.

    Raises:
        ValueError: Nếu cột không tồn tại hoặc có mã không thuộc bất kỳ nhóm nào trong mapping.
    """
    if isinstance(df_features, pd.DataFrame):
        if column not in df_features.columns:
            raise ValueError(f"Cột '{column}' không tồn tại trong dữ liệu đặc trưng.")
        series = df_features[column]
    elif isinstance(df_features, pd.Series):
        series = df_features
    elif isinstance(df_features, dict):
        if column not in df_features:
            raise ValueError(f"Cột '{column}' không tồn tại trong dữ liệu đặc trưng.")
        series = df_features[column]
    else:
        raise ValueError(f"Kiểu dữ liệu df_features không được hỗ trợ: {type(df_features)}")

    code_to_group: dict[int, str] = {}
    for group_name, codes in mapping.items():
        for code in codes:
            code_to_group[int(code)] = str(group_name)

    mapped_labels: list[str] = []
    for val in series:
        try:
            val_int = int(val)
        except (ValueError, TypeError):
            raise ValueError(f"Mã '{val}' trong cột '{column}' không phải số nguyên hợp lệ.")
        if val_int not in code_to_group:
            raise ValueError(f"Mã '{val}' trong cột '{column}' không thuộc bất kỳ nhóm nào trong mapping.")
        mapped_labels.append(code_to_group[val_int])

    return np.asarray(mapped_labels, dtype=object)


def group_metrics(
    groups: np.ndarray,
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int] | None = None,
    risk_flag: np.ndarray | list[bool] | None = None,
    positive_class: int = 2,
    min_pos: int = 30,
) -> list[dict[str, Any]]:
    """Tính các chỉ số theo từng nhóm nhân khẩu học kèm khoảng tin cậy Wilson.

    Args:
        groups: Mảng nhãn nhóm tương ứng cho từng mẫu.
        y_true: Mảng nhãn thực tế.
        y_pred: Mảng nhãn dự đoán từ mô hình (tùy chọn).
        risk_flag: Mảng boolean cờ cảnh báo nguy cơ cao (tùy chọn).
        positive_class: Nhãn của lớp dương (mặc định là 2: tiểu đường).
        min_pos: Ngưỡng số mẫu dương tối thiểu để nhóm không bị xem là mẫu nhỏ (mặc định 30).

    Returns:
        list[dict]: Danh sách từ điển thông tin từng nhóm:
            - group: Tên nhóm.
            - n: Tổng số mẫu trong nhóm.
            - n_pos: Số mẫu dương thực tế.
            - prevalence: Tỷ lệ dương kèm Wilson {"point", "low", "high"}.
            - recall: Độ nhạy kèm Wilson {"point", "low", "high"} (hoặc None).
            - precision: Độ chuẩn xác kèm Wilson {"point", "low", "high"} (hoặc None).
            - alert_rate: Tỷ lệ được cảnh báo kèm Wilson {"point", "low", "high"} (hoặc None).
            - small_sample: Cờ boolean (n_pos < min_pos).
    """
    groups_arr = np.asarray(groups)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred) if y_pred is not None else None
    risk_flag_arr = np.asarray(risk_flag, dtype=bool) if risk_flag is not None else None

    N = len(groups_arr)
    if len(y_true_arr) != N:
        raise ValueError(f"Độ dài groups ({N}) và y_true ({len(y_true_arr)}) không khớp.")
    if y_pred_arr is not None and len(y_pred_arr) != N:
        raise ValueError(f"Độ dài groups ({N}) và y_pred ({len(y_pred_arr)}) không khớp.")
    if risk_flag_arr is not None and len(risk_flag_arr) != N:
        raise ValueError(f"Độ dài groups ({N}) và risk_flag ({len(risk_flag_arr)}) không khớp.")

    unique_groups = list(dict.fromkeys(groups_arr))
    rows: list[dict[str, Any]] = []

    for g in unique_groups:
        mask = groups_arr == g
        n = int(np.sum(mask))
        y_t = y_true_arr[mask]
        is_pos = y_t == positive_class
        n_pos = int(np.sum(is_pos))
        small_sample = bool(n_pos < min_pos)

        # 1. Prevalence
        prev_point = float(n_pos / n) if n > 0 else 0.0
        prev_low, prev_high = wilson_interval(n_pos, n)
        prevalence = {"point": prev_point, "low": prev_low, "high": prev_high}

        # 2. Recall & Precision
        if y_pred_arr is not None:
            y_p = y_pred_arr[mask]
            pred_pos = y_p == positive_class
            tp = int(np.sum(is_pos & pred_pos))
            n_flagged = int(np.sum(pred_pos))

            rec_point = float(tp / n_pos) if n_pos > 0 else 0.0
            rec_low, rec_high = wilson_interval(tp, n_pos)
            recall = {"point": rec_point, "low": rec_low, "high": rec_high}

            prec_point = float(tp / n_flagged) if n_flagged > 0 else 0.0
            prec_low, prec_high = wilson_interval(tp, n_flagged)
            precision = {"point": prec_point, "low": prec_low, "high": prec_high}
        elif risk_flag_arr is not None:
            rf = risk_flag_arr[mask]
            tp = int(np.sum(is_pos & rf))
            n_flagged = int(np.sum(rf))

            rec_point = float(tp / n_pos) if n_pos > 0 else 0.0
            rec_low, rec_high = wilson_interval(tp, n_pos)
            recall = {"point": rec_point, "low": rec_low, "high": rec_high}

            prec_point = float(tp / n_flagged) if n_flagged > 0 else 0.0
            prec_low, prec_high = wilson_interval(tp, n_flagged)
            precision = {"point": prec_point, "low": prec_low, "high": prec_high}
        else:
            recall = None
            precision = None

        # 3. Alert rate
        if risk_flag_arr is not None:
            rf = risk_flag_arr[mask]
            n_alert = int(np.sum(rf))
            alert_point = float(n_alert / n) if n > 0 else 0.0
            alert_low, alert_high = wilson_interval(n_alert, n)
            alert_rate = {"point": alert_point, "low": alert_low, "high": alert_high}
        else:
            alert_rate = None

        rows.append(
            {
                "group": str(g),
                "n": n,
                "n_pos": n_pos,
                "prevalence": prevalence,
                "recall": recall,
                "precision": precision,
                "alert_rate": alert_rate,
                "small_sample": small_sample,
            }
        )

    return rows


def disparity(
    rows: list[dict[str, Any]],
    metric: str = "recall",
) -> dict[str, Any]:
    """Tính mức chênh lệch (disparity) lớn nhất giữa các nhóm hợp lệ (bỏ qua mẫu nhỏ).

    Args:
        rows: Danh sách số liệu các nhóm từ group_metrics.
        metric: Tên chỉ số cần tính chênh lệch (mặc định: 'recall').

    Returns:
        dict:
            - max_gap_pp: Chênh lệch lớn nhất tính theo điểm phần trăm (percentage points).
            - best_group: Tên nhóm có điểm số cao nhất.
            - worst_group: Tên nhóm có điểm số thấp nhất.
            - ignored_groups: Danh sách tên các nhóm bị bỏ qua do small_sample=True.
    """
    valid_entries: list[tuple[str, float]] = []
    ignored_groups: list[str] = []

    for r in rows:
        g_name = str(r["group"])
        if r.get("small_sample", False):
            ignored_groups.append(g_name)
            continue
        val_obj = r.get(metric)
        if val_obj is None:
            ignored_groups.append(g_name)
            continue
        val = val_obj["point"] if isinstance(val_obj, dict) else float(val_obj)
        valid_entries.append((g_name, float(val)))

    if not valid_entries:
        return {
            "max_gap_pp": 0.0,
            "best_group": None,
            "worst_group": None,
            "ignored_groups": ignored_groups,
        }

    best_group, best_val = max(valid_entries, key=lambda x: x[1])
    worst_group, worst_val = min(valid_entries, key=lambda x: x[1])
    max_gap_pp = float((best_val - worst_val) * 100.0)

    return {
        "max_gap_pp": max_gap_pp,
        "best_group": best_group,
        "worst_group": worst_group,
        "ignored_groups": ignored_groups,
    }


def fairness_report(
    df_features: pd.DataFrame,
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int] | None = None,
    risk_flag: np.ndarray | list[bool] | None = None,
    cfg: dict[str, Any] | None = None,
    min_pos: int | None = None,
) -> dict[str, Any]:
    """Tạo báo cáo công bằng cho các đặc trưng nhân khẩu học (Age, Income, Sex).

    Chạy cho hai lớp dương:
        - Tiểu đường (positive_class = 2)
        - Tiền tiểu đường (positive_class = 1, tham khảo)

    Args:
        df_features: DataFrame chứa các cột nhân khẩu học.
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        risk_flag: Cờ cảnh báo nguy cơ cao.
        cfg: Cấu hình chứa mục fairness (đọc từ evaluate.yaml).
        min_pos: Ngưỡng số mẫu dương tối thiểu (ghi đè cfg nếu có).

    Returns:
        dict: Báo cáo công bằng chứa thông tin theo từng thuộc tính nhân khẩu học.
    """
    if cfg is None:
        cfg_path = Path("configs/evaluate.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

    fairness_cfg = cfg.get("fairness", cfg) if isinstance(cfg, dict) else {}
    if min_pos is None:
        min_pos = int(fairness_cfg.get("min_pos", 30))

    col_configs: dict[str, tuple[str, dict[str, list[int]]]] = {
        "Age": (
            "age_groups",
            {
                "18-34": [1, 2, 3],
                "35-44": [4, 5],
                "45-54": [6, 7],
                "55-64": [8, 9],
                "65-74": [10, 11],
                "75+": [12, 13],
            },
        ),
        "Income": (
            "income_groups",
            {
                "<25k": [1, 2, 3, 4],
                "25-50k": [5, 6],
                ">=50k": [7, 8],
            },
        ),
        "Sex": (
            "sex_groups",
            {
                "Nữ": [0],
                "Nam": [1],
            },
        ),
    }

    report: dict[str, Any] = {"min_pos": min_pos}

    for col_name, (cfg_key, default_mapping) in col_configs.items():
        if col_name not in df_features.columns:
            continue
        mapping = fairness_cfg.get(cfg_key) or fairness_cfg.get(col_name) or default_mapping
        groups = group_codes(df_features, col_name, mapping)

        # Lớp 2: Tiểu đường
        rows_dia = group_metrics(
            groups, y_true, y_pred=y_pred, risk_flag=risk_flag, positive_class=2, min_pos=min_pos
        )
        disp_dia = disparity(rows_dia, metric="recall")

        # Lớp 1: Tiền tiểu đường (tham khảo)
        rows_pre = group_metrics(
            groups, y_true, y_pred=y_pred, risk_flag=risk_flag, positive_class=1, min_pos=min_pos
        )
        disp_pre = disparity(rows_pre, metric="recall")

        report[col_name] = {
            "rows": rows_dia,
            "disparity": disp_dia,
            "max_gap_pp": disp_dia["max_gap_pp"],
            "by_class": {
                "diabetes": {
                    "positive_class": 2,
                    "rows": rows_dia,
                    "disparity": disp_dia,
                },
                "prediabetes": {
                    "positive_class": 1,
                    "rows": rows_pre,
                    "disparity": disp_pre,
                },
            },
        }

    return report
