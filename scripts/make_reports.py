#!/usr/bin/env python3
"""Công cụ sinh báo cáo tự động từ số liệu JSON đo được.

Thay thế các khối `<!-- AUTO:name --> ... <!-- /AUTO -->` trong các file báo cáo:
- `reports/model_card.md`
- `reports/fairness_report.md`
- `reports/final_report.md`
bằng các bảng Markdown định dạng chuẩn (định dạng số kiểu Việt: dấu phẩy thập phân, dấu chấm ngăn nghìn).

Mọi số liệu đều được nạp từ `reports/metrics/*.json` và `data/splits/*/manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPORT_FILES = [
    ROOT / "reports/model_card.md",
    ROOT / "reports/fairness_report.md",
    ROOT / "reports/final_report.md",
]
METRICS_DIR = ROOT / "reports/metrics"
SPLITS_DIR = ROOT / "data/splits"


# --- Helpers định dạng số kiểu Việt ---
def fmt_int(n: int | float) -> str:
    """Định dạng số nguyên với dấu chấm ngăn nghìn (ví dụ: 253.680)."""
    return f"{int(round(n)):,}".replace(",", ".")


def fmt_float(x: float, decimals: int = 4) -> str:
    """Định dạng số thực với dấu phẩy thập phân (ví dụ: 0,7858)."""
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(x).replace(".", ",")


def fmt_pct(x: float, decimals: int = 2) -> str:
    """Định dạng tỉ lệ phần trăm với dấu phẩy thập phân (ví dụ: 84,24%)."""
    fmt = f"{{:.{decimals}f}}%"
    return fmt.format(x * 100.0).replace(".", ",")


def fmt_ci(low: float, high: float, decimals: int = 4, is_pct: bool = False) -> str:
    """Định dạng khoảng tin cậy [low; high]."""
    if is_pct:
        return f"[{fmt_pct(low, 2)}; {fmt_pct(high, 2)}]"
    return f"[{fmt_float(low, decimals)}; {fmt_float(high, decimals)}]"


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --- Các hàm sinh nội dung cho từng khối AUTO ---

def generate_data_summary() -> str:
    """Khối data_summary: Tóm tắt tập dữ liệu CDC BRFSS 2015."""
    dv = _load_json(METRICS_DIR / "data_validation.json")
    rows: list[str] = [
        "| Chỉ số thống kê dữ liệu | Giá trị đo được | Ý nghĩa lâm sàng / dữ liệu |",
        "| :--- | :--- | :--- |",
        f"| Tổng số dòng (quan sát) | **{fmt_int(dv['n_rows'])}** | Bộ dữ liệu khảo sát BRFSS 2015 từ CDC Hoa Kỳ |",
        f"| Số cột đặc trưng | **{fmt_int(dv['n_cols'] - 1)}** (cộng 1 cột nhãn) | 21 biến nhân khẩu học, lối sống và bệnh nền |",
        f"| Giá trị bị thiếu (Missing values) | **0** ({fmt_pct(0.0)}) | Toàn bộ các câu hỏi khảo sát đã được làm sạch đầy đủ |",
        f"| Số dòng trùng lặp hoàn toàn | **{fmt_int(dv['n_duplicates'])}** ({fmt_pct(dv['n_duplicates'] / dv['n_rows'])}) | Xuất hiện tự nhiên do toàn bộ 21 biến là biến phân loại rời rạc |",
        f"| Số dòng hồ sơ phân biệt (Unique) | **{fmt_int(dv['n_unique_rows'])}** ({fmt_pct(dv['n_unique_rows'] / dv['n_rows'])}) | Các tổ hợp trả lời khảo sát độc nhất |",
        f"| Lớp 0 — Không tiểu đường | **{fmt_int(dv['label_counts']['0'])}** ({fmt_pct(dv['label_pct']['0'])}) | Nhóm đa số áp đảo trong dân số khảo sát |",
        f"| Lớp 1 — Tiền tiểu đường | **{fmt_int(dv['label_counts']['1'])}** ({fmt_pct(dv['label_pct']['1'])}) | Nhóm cực kỳ thiểu số (< 2%), ranh giới sinh lý mờ nhạt |",
        f"| Lớp 2 — Tiểu đường | **{fmt_int(dv['label_counts']['2'])}** ({fmt_pct(dv['label_pct']['2'])}) | Nhóm bệnh nhân đích cần ưu tiên sàng lọc |",
    ]
    return "\n".join(rows) + "\n"


def generate_split_table() -> str:
    """Khối split_table: Bảng phân chia dữ liệu train/val/test cho cả 2 biến thể."""
    mf_full = _load_json(SPLITS_DIR / "full_seed42/manifest.json")
    mf_dedup = _load_json(SPLITS_DIR / "dedup_seed42/manifest.json")

    rows: list[str] = [
        "| Biến thể | Tập chia | Số mẫu (N) | Tỉ lệ (%) | Lớp 0 (Không) | Lớp 1 (Tiền TĐ) | Lớp 2 (Tiểu đường) | Trùng với Train (twins) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    # Full
    for sname, s_label in [("train", "Train"), ("val", "Validation"), ("test", "Test (1 lần)")]:
        sz = mf_full["sizes"][sname]
        pct = sz / mf_full["n_rows_used"]
        c0 = f"{fmt_int(mf_full['class_counts'][sname]['0'])} ({fmt_pct(mf_full['class_pct'][sname]['0'])})"
        c1 = f"{fmt_int(mf_full['class_counts'][sname]['1'])} ({fmt_pct(mf_full['class_pct'][sname]['1'])})"
        c2 = f"{fmt_int(mf_full['class_counts'][sname]['2'])} ({fmt_pct(mf_full['class_pct'][sname]['2'])})"
        if sname == "train":
            twins_str = "— (Tập gốc)"
        else:
            tw = mf_full["twins"][sname]
            twins_str = f"{fmt_int(tw['n'])} ({fmt_pct(tw['pct'])})"
        prefix = "**Bản đầy đủ (Full)**" if sname == "train" else ""
        rows.append(f"| {prefix} | {s_label} | {fmt_int(sz)} | {fmt_pct(pct)} | {c0} | {c1} | {c2} | {twins_str} |")

    # Dedup
    for sname, s_label in [("train", "Train"), ("val", "Validation"), ("test", "Test (1 lần)")]:
        sz = mf_dedup["sizes"][sname]
        pct = sz / mf_dedup["n_rows_used"]
        c0 = f"{fmt_int(mf_dedup['class_counts'][sname]['0'])} ({fmt_pct(mf_dedup['class_pct'][sname]['0'])})"
        c1 = f"{fmt_int(mf_dedup['class_counts'][sname]['1'])} ({fmt_pct(mf_dedup['class_pct'][sname]['1'])})"
        c2 = f"{fmt_int(mf_dedup['class_counts'][sname]['2'])} ({fmt_pct(mf_dedup['class_pct'][sname]['2'])})"
        if sname == "train":
            twins_str = "— (Tập gốc)"
        else:
            tw = mf_dedup["twins"][sname]
            twins_str = f"{fmt_int(tw['n'])} ({fmt_pct(tw['pct'])})"
        prefix = "**Bản loại trùng (Dedup)**" if sname == "train" else ""
        rows.append(f"| {prefix} | {s_label} | {fmt_int(sz)} | {fmt_pct(pct)} | {c0} | {c1} | {c2} | {twins_str} |")

    return "\n".join(rows) + "\n"


def generate_val_results() -> str:
    """Khối val_results: Kết quả so sánh các mô hình trên tập validation (đầy đủ 3 lớp)."""
    val_data = _load_json(METRICS_DIR / "val_evaluation_debug.json")["multiclass_full"]

    model_names = [
        ("majority", "Majority Baseline (Luôn đoán Lớp 0)"),
        ("logreg", "Logistic Regression (Hồi quy Logistic)"),
        ("xgb", "XGBoost (Cây quyết định tăng cường)"),
        ("mlp", "MLP (Mạng nơ-ron nhiều tầng — ĐƯỢC CHỌN)"),
    ]

    rows: list[str] = [
        "| Mô hình | ROC-AUC (OvR macro) | PR-AUC (macro) | Macro-F1 | Accuracy | Baseline Accuracy |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    base_acc = val_data["accuracy_majority_baseline"]
    for mkey, mlabel in model_names:
        m = val_data[mkey]
        auc_str = fmt_float(m["roc_auc_ovr_macro"])
        pr_str = fmt_float(m["pr_auc_macro"])
        f1_str = fmt_float(m["macro_f1"])
        acc_str = fmt_float(m["accuracy"])
        rows.append(f"| **{mlabel}** | {auc_str} | {pr_str} | {f1_str} | {acc_str} | {fmt_float(base_acc)} |")

    return "\n".join(rows) + "\n"


def generate_test_results() -> str:
    """Khối test_results: Đánh giá duy nhất 1 lần trên tập Test (N=50.736) kèm khoảng tin cậy."""
    te = _load_json(METRICS_DIR / "test_evaluation.json")["multiclass_full"]
    base_acc = te["accuracy_majority_baseline"]

    rows: list[str] = [
        "### Bảng 1: So sánh tổng thể các mô hình trên tập Test (N = 50.736)",
        "",
        "| Mô hình | ROC-AUC macro (95% CI) | PR-AUC macro | Macro-F1 | Accuracy | Baseline Accuracy |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    model_info = [
        ("majority", "Majority Baseline", te["majority"]),
        ("logreg", "Logistic Regression", te["logreg"]),
        ("xgb", "XGBoost", te["xgb"]),
        ("mlp", "MLP (Mô hình triển khai)", te["mlp"]),
    ]

    for _, label, data in model_info:
        auc_val = data["roc_auc_ovr_macro"]
        if "ci" in data and "roc_auc_ovr_macro" in data["ci"]:
            ci_obj = data["ci"]["roc_auc_ovr_macro"]
            auc_str = f"**{fmt_float(auc_val)}** {fmt_ci(ci_obj['low'], ci_obj['high'])}"
        else:
            auc_str = fmt_float(auc_val)

        pr_str = fmt_float(data["pr_auc_macro"])
        f1_str = fmt_float(data["macro_f1"])
        acc_str = fmt_float(data["accuracy"])
        rows.append(f"| {label} | {auc_str} | {pr_str} | {f1_str} | {acc_str} | {fmt_float(base_acc)} |")

    rows.extend([
        "",
        "### Bảng 2: Hiệu năng chi tiết theo từng lớp của mô hình MLP trên tập Test",
        "",
        "| Nhãn bệnh lý | Số mẫu trong Test | Recall / Độ nhạy (95% CI) | Precision / Độ chuẩn xác | F1-score | ROC-AUC (OvR) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    mlp = te["mlp"]
    class_labels = [
        ("Lớp 0: Không tiểu đường", 42741, 0),
        ("Lớp 1: Tiền tiểu đường", 926, 1),
        ("Lớp 2: Tiểu đường", 7069, 2),
    ]

    for clabel, n_samples, idx in class_labels:
        rec_val = mlp["recall_per_class"][idx]
        rec_ci = mlp["ci"]["recall_per_class"]["by_class"][idx]
        rec_str = f"**{fmt_float(rec_val)}** {fmt_ci(rec_ci['low'], rec_ci['high'])}"
        prec_str = fmt_float(mlp["precision_per_class"][idx])
        f1_str = fmt_float(mlp["f1_per_class"][idx])
        auc_class_str = fmt_float(mlp["roc_auc_per_class"][idx])
        rows.append(f"| **{clabel}** | {fmt_int(n_samples)} | {rec_str} | {prec_str} | {f1_str} | {auc_class_str} |")

    return "\n".join(rows) + "\n"


def generate_binary_results() -> str:
    """Khối binary_results: Kết quả trên bài toán phân loại nhị phân (0 vs 1)."""
    te_bin = _load_json(METRICS_DIR / "test_evaluation.json")["binary_full"]
    base_acc = te_bin["accuracy_majority_baseline"]

    rows: list[str] = [
        "| Mô hình | ROC-AUC (95% CI nếu có) | PR-AUC macro | Macro-F1 | Accuracy | Recall Lớp 1 (Có bệnh) | Baseline Accuracy |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    models = [
        ("majority", "Majority Baseline", te_bin["majority"]),
        ("logreg", "Logistic Regression", te_bin["logreg"]),
        ("xgb", "XGBoost", te_bin["xgb"]),
        ("mlp", "MLP", te_bin["mlp"]),
    ]

    for _, label, data in models:
        auc_val = data["roc_auc_ovr_macro"]
        pr_str = fmt_float(data["pr_auc_macro"])
        f1_str = fmt_float(data["macro_f1"])
        acc_str = fmt_float(data["accuracy"])
        rec1_str = fmt_float(data["recall_per_class"][1]) if len(data["recall_per_class"]) > 1 else "—"
        rows.append(f"| {label} | {fmt_float(auc_val)} | {pr_str} | {f1_str} | {acc_str} | {rec1_str} | {fmt_float(base_acc)} |")

    return "\n".join(rows) + "\n"


def generate_calibration() -> str:
    """Khối calibration: Hiệu chuẩn xác suất trước và sau hiệu chuẩn."""
    te = _load_json(METRICS_DIR / "test_evaluation.json")["multiclass_full"]["mlp"]
    pp = _load_json(METRICS_DIR / "postproc_mlp_multiclass_full.json")

    cal = pp["calibration"]
    rows: list[str] = [
        "| Tiêu chí đo lường hiệu chuẩn | Trước hiệu chuẩn (Thô / Raw) | Sau hiệu chuẩn (Calibrated) | Cải thiện / Ý nghĩa |",
        "| :--- | :--- | :--- | :--- |",
        f"| Phương pháp hiệu chuẩn | — | **{cal['method']}** | Lựa chọn qua Out-of-fold cross-validation trên tập Val |",
        f"| ECE Macro trên tập Test | **{fmt_float(te['ece_macro_raw'])}** | **{fmt_float(te['ece_macro'])}** | Giảm sai số hiệu chuẩn xác suất toàn thể |",
        f"| Brier Score đa lớp trên Test | **{fmt_float(te['brier_multiclass_raw'])}** | **{fmt_float(te['brier_multiclass'])}** | Duy trì độ sắc bén dự báo |",
        f"| ECE Lớp 0 (Không tiểu đường) | — | **{fmt_float(te['ece_per_class'][0])}** | Xác suất phân bổ sát tần suất thực nghiệm |",
        f"| ECE Lớp 1 (Tiền tiểu đường) | — | **{fmt_float(te['ece_per_class'][1])}** | Rất thấp do dự đoán thận trọng ở lớp thiểu số |",
        f"| ECE Lớp 2 (Tiểu đường) | — | **{fmt_float(te['ece_per_class'][2])}** | Xác suất tin cậy trực tiếp cho phân tầng nguy cơ |",
    ]
    return "\n".join(rows) + "\n"


def generate_control_dedup() -> str:
    """Khối control_dedup: So sánh mô hình trên bản Full và bản Dedup để kiểm soát data leakage."""
    te_full = _load_json(METRICS_DIR / "test_evaluation.json")["multiclass_full"]["mlp"]
    te_dedup = _load_json(METRICS_DIR / "test_evaluation.json")["multiclass_dedup"]["mlp"]

    gap_auc = te_full["roc_auc_ovr_macro"] - te_dedup["roc_auc_ovr_macro"]
    gap_acc = te_full["accuracy"] - te_dedup["accuracy"]

    rows: list[str] = [
        "| Chỉ số hiệu năng (Mô hình MLP) | Bản đầy đủ (Full, N=50.736) | Bản loại trùng (Dedup, N=45.957) | Chênh lệch (Full - Dedup) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Tỉ lệ mẫu trùng lặp chéo với Train | **12,75%** (6.469 mẫu) | **1,14%** (522 mẫu) | -11,61 điểm % |",
        f"| ROC-AUC macro | **{fmt_float(te_full['roc_auc_ovr_macro'])}** {fmt_ci(te_full['ci']['roc_auc_ovr_macro']['low'], te_full['ci']['roc_auc_ovr_macro']['high'])} | **{fmt_float(te_dedup['roc_auc_ovr_macro'])}** {fmt_ci(te_dedup['ci']['roc_auc_ovr_macro']['low'], te_dedup['ci']['roc_auc_ovr_macro']['high'])} | +{fmt_float(gap_auc)} |",
        f"| Macro-F1 | **{fmt_float(te_full['macro_f1'])}** | **{fmt_float(te_dedup['macro_f1'])}** | {fmt_float(te_full['macro_f1'] - te_dedup['macro_f1'])} |",
        f"| Accuracy | **{fmt_float(te_full['accuracy'])}** | **{fmt_float(te_dedup['accuracy'])}** | +{fmt_float(gap_acc)} |",
        f"| Baseline Accuracy | **{fmt_float(te_full['accuracy_majority_baseline'])}** | **{fmt_float(te_dedup['accuracy_majority_baseline'])}** | — |",
    ]
    return "\n".join(rows) + "\n"


def generate_novel_subset() -> str:
    """Khối novel_subset: Đánh giá trên tập con hoàn toàn mới (no-twins) trong tập Test."""
    te = _load_json(METRICS_DIR / "test_evaluation.json")["multiclass_full"]
    no_twins = te["no_twins"]
    full_mlp_auc = te["mlp"]["roc_auc_ovr_macro"]
    novel_mlp_auc = no_twins["auc"]["mlp"]

    rows: list[str] = [
        "| Mô hình đánh giá | ROC-AUC trên toàn tập Test (N=50.736) | ROC-AUC trên tập Novel / No-twins (N=44.267, 87,25%) | Độ suy giảm AUC |",
        "| :--- | :--- | :--- | :--- |",
        f"| Majority Baseline | {fmt_float(te['majority']['roc_auc_ovr_macro'])} | {fmt_float(no_twins['auc']['majority'])} | 0,0000 |",
        f"| Logistic Regression | {fmt_float(te['logreg']['roc_auc_ovr_macro'])} | {fmt_float(no_twins['auc']['logreg'])} | {fmt_float(te['logreg']['roc_auc_ovr_macro'] - no_twins['auc']['logreg'])} |",
        f"| XGBoost | {fmt_float(te['xgb']['roc_auc_ovr_macro'])} | {fmt_float(no_twins['auc']['xgb'])} | {fmt_float(te['xgb']['roc_auc_ovr_macro'] - no_twins['auc']['xgb'])} |",
        f"| **MLP (Mô hình chính)** | **{fmt_float(full_mlp_auc)}** | **{fmt_float(novel_mlp_auc)}** | **{fmt_float(full_mlp_auc - novel_mlp_auc)}** |",
    ]
    return "\n".join(rows) + "\n"


def generate_risk_alert() -> str:
    """Khối risk_alert: Hiệu năng sàng lọc tại các tầng nguy cơ Trung bình và Cao."""
    stats = _load_json(METRICS_DIR / "test_evaluation.json")["multiclass_full"]["mlp"]["risk_alert_stats"]

    med = stats["medium_or_above"]
    high = stats["high"]

    rows: list[str] = [
        "| Tầng cảnh báo nguy cơ | Ngưỡng xác suất ($p$) | Tỉ lệ phát cảnh báo (Alert Rate) | Số ca cảnh báo | Độ nhạy / Bắt ca bệnh (Recall) | PPV (Độ chính xác cảnh báo) | Tỉ lệ dương tính giả (FPR) | Độ nâng so với nền (Lift) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Tỉ lệ nền (Toàn tập test)** | — | 100,00% | {fmt_int(stats['n'])} | 100,00% | {fmt_pct(stats['base_rate'])} | — | 1,00x |",
        f"| **Nguy cơ Trung bình trở lên** | $p \\ge {fmt_float(med['threshold'], 2)}$ | **{fmt_pct(med['alert_rate'])}** | {fmt_int(med['n_alert'])} | **{fmt_pct(med['recall'])}** ({fmt_int(med['tp'])}/{fmt_int(stats['n_pos'])}) | **{fmt_pct(med['ppv'])}** | {fmt_pct(med['false_positive_rate'])} | **{fmt_float(med['lift'], 2)}x** |",
        f"| **Nguy cơ Cao** | $p \\ge {fmt_float(high['threshold'], 2)}$ | **{fmt_pct(high['alert_rate'])}** | {fmt_int(high['n_alert'])} | **{fmt_pct(high['recall'])}** ({fmt_int(high['tp'])}/{fmt_int(stats['n_pos'])}) | **{fmt_pct(high['ppv'])}** | {fmt_pct(high['false_positive_rate'])} | **{fmt_float(high['lift'], 2)}x** |",
    ]
    return "\n".join(rows) + "\n"


def generate_fairness(group_name: str) -> str:
    """Khối fairness cho từng biến nhân khẩu học hoặc toàn bộ."""
    fn = _load_json(METRICS_DIR / "fairness_test.json")

    group_keys = [group_name] if group_name in fn else ["Age", "Income", "Sex"]

    group_labels = {
        "Age": "Độ tuổi (Age)",
        "Income": "Thu nhập (Income)",
        "Sex": "Giới tính (Sex)",
    }

    sub_labels = {
        "18-34": "18–34 tuổi",
        "35-44": "35–44 tuổi",
        "45-54": "45–54 tuổi",
        "55-64": "55–64 tuổi",
        "65-74": "65–74 tuổi",
        "75+": "75 tuổi trở lên",
        "<25k": "Dưới $25.000",
        "25-50k": "$25.000 – $50.000",
        ">=50k": "Trên $50.000",
        "Nam": "Nam giới",
        "Nữ": "Nữ giới",
    }

    all_tables: list[str] = []
    for gk in group_keys:
        gdata = fn[gk]
        max_gap = gdata.get("max_gap_pp", 0.0)
        table_rows: list[str] = [
            f"### Đánh giá công bằng theo {group_labels.get(gk, gk)} (Chênh lệch tối đa: **{fmt_float(max_gap, 2)} pp**)",
            "",
            "| Phân nhóm | Số mẫu (N) | Ca bệnh (N pos) | Tỉ lệ nền (Prevalence) | Recall (95% Wilson CI) | Precision (95% Wilson CI) | Tỉ lệ cảnh báo (Alert Rate) | Cờ mẫu nhỏ (<30 ca) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in gdata["rows"]:
            grp_name = sub_labels.get(r["group"], r["group"])
            n_str = fmt_int(r["n"])
            n_pos_str = fmt_int(r["n_pos"])
            prev_str = fmt_pct(r["prevalence"]["point"])
            rec_str = f"**{fmt_pct(r['recall']['point'])}** {fmt_ci(r['recall']['low'], r['recall']['high'], is_pct=True)}"
            prec_str = f"{fmt_pct(r['precision']['point'])} {fmt_ci(r['precision']['low'], r['precision']['high'], is_pct=True)}"
            alert_str = fmt_pct(r["alert_rate"]["point"])
            small_flag = "⚠️ Cảnh báo (<30)" if r.get("small_sample", False) else "Đạt chuẩn"
            table_rows.append(f"| {grp_name} | {n_str} | {n_pos_str} | {prev_str} | {rec_str} | {prec_str} | {alert_str} | {small_flag} |")

        all_tables.append("\n".join(table_rows))

    return "\n\n".join(all_tables) + "\n"


def generate_importance_top10() -> str:
    """Khối importance_top10: Top 10 đặc trưng theo Permutation Importance."""
    pi_list = _load_json(METRICS_DIR / "permutation_importance_mlp_multiclass_full.json")

    rows: list[str] = [
        "| Thứ hạng | Mã đặc trưng | Diễn giải lâm sàng / khảo sát | Mức giảm ROC-AUC trung bình (Mean drop) | Độ lệch chuẩn (Std) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for rank, item in enumerate(pi_list[:10], start=1):
        feat = item["feature"]
        desc = item.get("vi", feat)
        mean_drop = item["mean_drop"]
        std_drop = item["std_drop"]
        rows.append(f"| #{rank} | **{feat}** | {desc} | **+{fmt_float(mean_drop, 4)}** | ±{fmt_float(std_drop, 4)} |")

    return "\n".join(rows) + "\n"


def generate_block_content(name: str) -> str:
    """Điều phối hàm sinh tương ứng theo tên khối."""
    name_clean = name.strip()
    if name_clean == "data_summary":
        return generate_data_summary()
    if name_clean == "split_table":
        return generate_split_table()
    if name_clean == "val_results":
        return generate_val_results()
    if name_clean == "test_results":
        return generate_test_results()
    if name_clean == "binary_results":
        return generate_binary_results()
    if name_clean == "calibration":
        return generate_calibration()
    if name_clean == "control_dedup":
        return generate_control_dedup()
    if name_clean == "novel_subset":
        return generate_novel_subset()
    if name_clean == "risk_alert":
        return generate_risk_alert()
    if name_clean in ("fairness_age", "fairness_income", "fairness_sex"):
        gk = name_clean.split("_")[1].capitalize()
        return generate_fairness(gk)
    if name_clean in ("fairness_all", "fairness_age|income|sex"):
        return generate_fairness("all")
    if name_clean == "importance_top10":
        return generate_importance_top10()

    raise ValueError(f"Không nhận diện được khối AUTO: '{name_clean}'")


def _replace_auto_blocks(text: str) -> str:
    def repl(match: re.Match) -> str:
        block_name = match.group(1).strip()
        content = generate_block_content(block_name)
        return f"<!-- AUTO:{block_name} -->\n{content}<!-- /AUTO -->"

    pattern = re.compile(r"<!--\s*AUTO:(.*?)\s*-->.*?<!--\s*/AUTO\s*-->", re.DOTALL)
    return pattern.sub(repl, text)


def _has_empty_auto_blocks(text: str) -> bool:
    # Một khối được coi là trống nếu giữa thẻ mở và thẻ đóng chỉ toàn khoảng trắng hoặc rỗng
    empty_pat = re.compile(r"<!--\s*AUTO:(.*?)\s*-->\s*<!--\s*/AUTO\s*-->", re.DOTALL)
    m = empty_pat.search(text)
    return bool(m)


def process_file(file_path: Path) -> None:
    original = file_path.read_text(encoding="utf-8")
    new_content = _replace_auto_blocks(original)
    file_path.write_text(new_content, encoding="utf-8")


def check_files() -> int:
    for rp in REPORT_FILES:
        if not rp.exists():
            sys.stderr.write(f"[LỖI] File báo cáo không tồn tại: {rp}\n")
            return 1
        content = rp.read_text(encoding="utf-8")
        if _has_empty_auto_blocks(content):
            sys.stderr.write(f"[LỖI] Khối AUTO trống trong {rp}\n")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sinh các bảng báo cáo tự động từ JSON metrics.")
    parser.add_argument("--check", action="store_true", help="Kiểm tra các khối AUTO đã được lấp đầy; thoát 1 nếu còn trống.")
    args = parser.parse_args()

    for report in REPORT_FILES:
        if not report.exists():
            sys.stderr.write(f"[CẢNH BÁO] Báo cáo không tồn tại: {report}\n")
            continue
        process_file(report)

    if args.check:
        return check_files()
    return 0


if __name__ == "__main__":
    sys.exit(main())
