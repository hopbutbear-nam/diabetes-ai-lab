"""Script đánh giá mô hình trên tập validation hoặc test (chỉ chạy test 1 lần duy nhất).

Quy trình:
  1. Đọc cấu hình từ configs/evaluate.yaml.
  2. Nếu --split test: tính SHA-256 các artifact và xin khóa test (sys.exit(2) nếu đã có khóa).
  3. Nạp dữ liệu qua load_experiment_data(..., include_test=True khi test, False khi val).
  4. Đánh giá tất cả mô hình trong test.runs:
     - Tính proba_cal (calibrated=True), proba_raw (calibrated=False), y_pred qua predict_labels.
     - Tính compute_all, ECE/Brier trước sau.
     - Bootstrap khoảng tin cậy 95% (n=1000).
     - Thống kê cảnh báo rủi ro risk_alert_stats (A7).
  5. Riêng multiclass_full:
     - Đo số liệu tập con không bản sinh đôi (~twin_mask).
     - Mô hình tốt nhất: tạo báo cáo công bằng fairness_report và 6 biểu đồ PNG.
  6. Ghi reports/metrics/test_evaluation.json, fairness_test.json, và nhật ký experiment_log.md.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

# Đảm bảo import được src
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diabetes_ai.data.load import load_raw
from diabetes_ai.data.split import load_split, twin_mask
from diabetes_ai.evaluation.confidence_intervals import bootstrap_multi
from diabetes_ai.evaluation.fairness import fairness_report
from diabetes_ai.evaluation.metrics import (
    brier_multiclass,
    compute_all,
    ece_ovr,
    majority_baseline_accuracy,
    roc_auc_ovr,
)
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
    risk_alert_stats,
    write_json,
)
from diabetes_ai.features.build_features import load_experiment_data
from diabetes_ai.models.registry import load_run

METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
LOG_PATH = ROOT / "docs" / "experiment_log.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình trên tập validation hoặc test (chỉ test 1 lần duy nhất)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluate.yaml",
        help="Đường dẫn tới file cấu hình đánh giá (mặc định configs/evaluate.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Tập dữ liệu đánh giá: 'test' (mặc định) hoặc 'val' (để gỡ lỗi không khóa).",
    )
    parser.add_argument(
        "--force-reeval",
        action="store_true",
        help="Ép buộc đánh giá lại tập test khi đã có khóa trước đó.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Lý do ép buộc đánh giá lại (bắt buộc khi dùng --force-reeval).",
    )
    parser.add_argument(
        "--boot-n",
        type=int,
        default=None,
        help="Ghi đè số lượt bootstrap n (mặc định đọc từ evaluate.yaml).",
    )
    return parser.parse_args()


def _collect_hashes(runs: list[str]) -> dict[str, str]:
    """Tính sha256 cho model.*, scaler.json và postproc_<run>.json cho các run."""
    hashes: dict[str, str] = {}
    for run_name in runs:
        run_dir = MODELS_DIR / run_name
        if run_dir.is_dir():
            for p in run_dir.iterdir():
                if p.is_file() and (p.name.startswith("model.") or p.name == "scaler.json"):
                    hashes[f"{run_name}/{p.name}"] = file_sha256(p)
        postproc_file = METRICS_DIR / f"postproc_{run_name}.json"
        if postproc_file.is_file():
            hashes[f"metrics/{postproc_file.name}"] = file_sha256(postproc_file)
    return hashes


def _append_to_log(entries: list[dict[str, Any]]) -> None:
    """Ghi bổ sung các dòng đánh giá vào docs/experiment_log.md."""
    if not LOG_PATH.is_file():
        return
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for e in entries:
        row = (
            f"| {e['date']} | {e['model']} | {e['config']} | "
            f"{e['auc']} | {e['macro_f1']} | {e['note']} |\n"
        )
        new_lines.append(row)

    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.extend(new_lines)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> int:
    args = _parse_args()
    cfg_path = ROOT / args.config
    if not cfg_path.is_file():
        print(f"Không tìm thấy file cấu hình: {cfg_path}", file=sys.stderr)
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_cfg = cfg.get("test", {})
    run_groups = test_cfg.get("runs", ["multiclass_full", "binary_full", "multiclass_dedup"])
    model_keys = ["majority", "logreg", "xgb", "mlp"]
    all_runs = [f"{mk}_{group}" for group in run_groups for mk in model_keys]

    # Nạp thông tin chọn mô hình
    model_sel_path = METRICS_DIR / "model_selection.json"
    model_selection: dict[str, Any] = {}
    if model_sel_path.is_file():
        with open(model_sel_path, "r", encoding="utf-8") as f:
            model_selection = json.load(f)

    is_test = args.split == "test"
    lock_path = ROOT / test_cfg.get("lock_file", "reports/metrics/test_lock.json")

    # 1. Kiểm tra khóa nếu là test split
    if is_test:
        print("Đang tính SHA-256 các artifact và kiểm tra khóa tập test...")
        hashes = _collect_hashes(all_runs)
        try:
            acquire_test_lock(
                path=lock_path,
                runs=all_runs,
                model_hashes=hashes,
                force=args.force_reeval,
                reason=args.reason,
            )
            print(f"Đã xác nhận khóa tập test tại: {lock_path}")
        except RuntimeError as err:
            print(f"Lỗi khóa tập test: {err}", file=sys.stderr)
            sys.exit(2)

    boot_cfg = cfg.get("bootstrap", {})
    boot_n = args.boot_n if args.boot_n is not None else int(boot_cfg.get("n", 1000))
    boot_ci = float(boot_cfg.get("ci", 0.95))
    boot_seed = int(boot_cfg.get("seed", 42))

    df_raw = load_raw()
    split_full = load_split("full", seed=42)

    results: dict[str, Any] = {}
    fairness_output: dict[str, Any] = {}
    log_entries: list[dict[str, Any]] = []
    today = datetime.date.today().isoformat()

    for group_name in run_groups:
        parts = group_name.split("_")
        task = parts[0]
        variant = parts[1]

        print(f"\n==================================================")
        print(f"Đánh giá nhóm: {group_name} | split={args.split} | task={task} | variant={variant}")
        print(f"==================================================")

        # Nạp dữ liệu thực nghiệm
        if is_test:
            data = load_experiment_data(task=task, variant=variant, include_test=True)
            X = data.X_test
            y = data.y_test
            idx = data.idx_test
        else:
            data = load_experiment_data(task=task, variant=variant, include_test=False)
            X = data.X_val
            y = data.y_val
            idx = data.idx_val

        assert X is not None and y is not None and idx is not None

        n_samples = len(y)
        acc_majority = float(majority_baseline_accuracy(y))
        best_run_for_group = model_selection.get(group_name, {}).get("best", f"mlp_{group_name}")

        group_result: dict[str, Any] = {
            "n": int(n_samples),
            "accuracy_majority_baseline": acc_majority,
            "best_model": best_run_for_group,
        }

        # Tính mask tập con không bản sinh đôi nếu là multiclass_full
        no_twin_mask = None
        n_no_twins = 0
        if group_name == "multiclass_full":
            is_twin = twin_mask(df_raw, split_full.train, idx)
            no_twin_mask = ~is_twin
            n_no_twins = int(np.sum(no_twin_mask))
            group_result["no_twins"] = {
                "n": n_no_twins,
                "pct": round(float(n_no_twins / n_samples), 4),
                "auc": {},
            }

        # Xác định các mô hình cần bootstrap để tiết kiệm thời gian
        # multiclass_full: logreg, xgb, mlp
        # binary_full & multiclass_dedup: mô hình tốt nhất
        models_to_bootstrap: set[str] = set()
        if group_name == "multiclass_full":
            models_to_bootstrap = {"logreg", "xgb", "mlp"}
        else:
            best_mk = best_run_for_group.split("_")[0]
            models_to_bootstrap = {best_mk}

        for mk in model_keys:
            run_name = f"{mk}_{group_name}"
            print(f"\n--- Đang đánh giá run: {run_name} ---")
            run = load_run(run_name)

            proba_cal = run.predict_proba_scaled(X, calibrated=True)
            proba_raw = run.predict_proba_scaled(X, calibrated=False)
            y_pred = run.predict_labels(proba_cal)

            metrics = compute_all(y, proba_cal, y_pred=y_pred)

            # ECE và Brier thô (chưa hiệu chuẩn) để so sánh trước/sau
            ece_raw_info = ece_ovr(y, proba_raw)
            metrics["ece_macro_raw"] = float(ece_raw_info["macro"])
            metrics["brier_multiclass_raw"] = float(brier_multiclass(y, proba_raw))

            # Chỉ số nguy cơ (A7)
            if task == "multiclass":
                risk_index = proba_cal[:, 2] + 0.5 * proba_cal[:, 1]
                pos_cls = 2
            else:
                col = 1 if proba_cal.shape[1] > 1 else 0
                risk_index = proba_cal[:, col]
                pos_cls = 1

            risk_stats = risk_alert_stats(
                y_true=y,
                risk_index=risk_index,
                thresholds={"medium_or_above": 0.10, "high": 0.25},
                positive_class=pos_cls,
            )
            metrics["risk_alert_stats"] = risk_stats

            # Bootstrap khoảng tin cậy
            ci_data = None
            if mk in models_to_bootstrap and mk != "majority":
                print(f"  Chạy bootstrap (n={boot_n}, ci={boot_ci})...")
                ci_data = bootstrap_multi(
                    y_true=y,
                    proba=proba_cal,
                    y_pred=y_pred,
                    n=boot_n,
                    ci=boot_ci,
                    seed=boot_seed,
                )
                metrics["ci"] = ci_data
                auc_ci = ci_data["roc_auc_ovr_macro"]
                print(
                    f"  AUC: {metrics['roc_auc_ovr_macro']:.4f} "
                    f"[{auc_ci['low']:.4f}, {auc_ci['high']:.4f}]"
                )
            else:
                print(f"  AUC: {metrics.get('roc_auc_ovr_macro')}")

            # Tập con không bản sinh đôi
            if no_twin_mask is not None:
                auc_no_twin = float(roc_auc_ovr(y[no_twin_mask], proba_cal[no_twin_mask]))
                metrics["no_twins"] = {
                    "n": n_no_twins,
                    "roc_auc_ovr_macro": auc_no_twin,
                }
                group_result["no_twins"]["auc"][mk] = auc_no_twin
                print(f"  AUC trên tập con không bản sinh đôi (n={n_no_twins}): {auc_no_twin:.4f}")

            group_result[mk] = metrics

            # Tạo báo cáo công bằng và 6 hình ảnh cho mô hình tốt nhất của multiclass_full
            if group_name == "multiclass_full" and run_name == best_run_for_group:
                print(f"\n>> Tạo báo cáo công bằng và biểu đồ cho mô hình tốt nhất ({run_name})...")
                df_sub = df_raw.loc[idx].reset_index(drop=True)
                risk_flag = risk_index >= 0.25
                fairness_data = fairness_report(
                    df_features=df_sub,
                    y_true=y,
                    y_pred=y_pred,
                    risk_flag=risk_flag,
                    cfg=cfg,
                )
                fairness_output = fairness_data

                # Vẽ 6 biểu đồ vào reports/figures/
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                class_names = ["Không", "Tiền ĐTĐ", "ĐTĐ"]

                # 1. roc_ovr
                plot_roc_ovr(
                    y,
                    proba_cal,
                    class_names,
                    FIGURES_DIR / f"roc_ovr_{run_name}.png",
                )
                # 2. pr_ovr
                plot_pr_ovr(
                    y,
                    proba_cal,
                    class_names,
                    FIGURES_DIR / f"pr_ovr_{run_name}.png",
                )
                # 3. confusion
                cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
                plot_confusion(
                    cm,
                    class_names,
                    FIGURES_DIR / f"confusion_{run_name}.png",
                )
                # 4. reliability
                plot_reliability(
                    y,
                    proba_cal,
                    class_names,
                    FIGURES_DIR / f"reliability_{run_name}.png",
                    proba_before=proba_raw,
                )
                # 5. learning_curve (từ fit_info.history nếu có)
                history = run.meta.get("fit_info", {}).get("history")
                if history:
                    plot_learning_curve(
                        history,
                        FIGURES_DIR / f"learning_curve_{run_name}.png",
                    )
                # 6. fairness_recall
                plot_fairness_bars(
                    fairness_data,
                    metric="recall",
                    path=FIGURES_DIR / f"fairness_recall_{run_name}.png",
                )
                print(f">> Đã lưu 6 biểu đồ tại: {FIGURES_DIR}")

            # Chuẩn bị dòng nhật ký thí nghiệm
            if is_test:
                is_logged_model = False
                if group_name == "multiclass_full":
                    is_logged_model = True
                elif run_name == best_run_for_group:
                    is_logged_model = True

                if is_logged_model:
                    auc_val = metrics.get("roc_auc_ovr_macro")
                    auc_str = f"{auc_val:.3f}" if isinstance(auc_val, float) else "—"
                    f1_val = metrics.get("macro_f1")
                    f1_str = f"{f1_val:.3f}" if isinstance(f1_val, float) else "—"

                    note_parts = ["TEST (1 lần)", group_name]
                    if run_name == best_run_for_group:
                        note_parts.append("mô hình tốt nhất")
                    if ci_data:
                        auc_ci = ci_data["roc_auc_ovr_macro"]
                        note_parts.append(f"CI AUC=[{auc_ci['low']:.3f}, {auc_ci['high']:.3f}]")
                    if mk == "majority":
                        note_parts.append("baseline")

                    log_entries.append({
                        "date": today,
                        "model": mk,
                        "config": "configs/evaluate.yaml",
                        "auc": auc_str,
                        "macro_f1": f1_str,
                        "note": "; ".join(note_parts),
                    })

        results[group_name] = group_result

    # Ghi kết quả ra đĩa
    if is_test:
        eval_path = METRICS_DIR / "test_evaluation.json"
        write_json(results, eval_path)
        print(f"\nĐã ghi kết quả đánh giá tập test: {eval_path}")

        if fairness_output:
            fairness_path = METRICS_DIR / "fairness_test.json"
            write_json(fairness_output, fairness_path)
            print(f"Đã ghi báo cáo công bằng tập test: {fairness_path}")

        if log_entries:
            _append_to_log(log_entries)
            print(f"Đã cập nhật nhật ký thí nghiệm: {LOG_PATH}")
    else:
        debug_path = METRICS_DIR / "val_evaluation_debug.json"
        write_json(results, debug_path)
        print(f"\n[Gỡ lỗi] Đã ghi kết quả validation debug: {debug_path}")

    print("\nHoàn tất đánh giá thành công!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
