"""Hiệu chuẩn xác suất và chọn hệ số ngưỡng trên tập validation.

Quy trình:
  1. Nạp từng run đã huấn luyện.
  2. Dự đoán xác suất gốc trên validation (không hiệu chuẩn).
  3. Chọn phương pháp hiệu chuẩn tốt nhất (out-of-fold).
  4. Fit calibrator trên toàn validation, lưu models/<run>/calibrator.joblib.
  5. Chọn hệ số ngưỡng tối ưu trên OOF đã hiệu chuẩn.
  6. Ghi reports/metrics/postproc_<run>.json.
  7. Chọn mô hình tốt nhất theo quy tắc A6 (AUC OvR + simplicity_margin).
  8. Ghi reports/metrics/model_selection.json.
  9. Thêm dòng nhật ký vào docs/experiment_log.md.

Lưu ý:
  - Chỉ dùng validation; tập test KHÔNG được nạp.
  - majority không hiệu chuẩn (ghi postproc rỗng).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Thêm src và root vào sys.path để import được diabetes_ai và evaluation
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # cho package evaluation/ ở root

from diabetes_ai.data.load import load_raw
from diabetes_ai.data.preprocess import encode_target
from diabetes_ai.data.schema import FEATURES
from diabetes_ai.data.split import load_split
from diabetes_ai.evaluation.calibration import (
    fit_calibrator,
    oof_calibrated,
    save_calibrator,
    select_calibration_method,
)
from diabetes_ai.evaluation.metrics import brier_multiclass, ece_ovr, roc_auc_ovr
from diabetes_ai.models.registry import load_run
from evaluation.thresholds import apply_scale, select_binary_threshold, select_scale_factors

METRICS_DIR = ROOT / "reports" / "metrics"
MODELS_DIR = ROOT / "models"
LOG_PATH = ROOT / "docs" / "experiment_log.md"
CFG_PATH = ROOT / "configs" / "evaluate.yaml"


def _load_config() -> dict:
    """Nạp cấu hình từ configs/evaluate.yaml."""
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_val_data(task: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Nạp X_val và y_val từ split (chỉ validation, không nạp test).

    Args:
        task: Loại bài toán ('multiclass' hoặc 'binary').
        variant: Biến thể dữ liệu ('full' hoặc 'dedup').

    Returns:
        Tuple (X_val_scaled_placeholder, y_val) — X chưa scale (sẽ scale qua run.scaler).
    """
    split = load_split(variant, seed=42)
    df = load_raw()
    X_all = df[FEATURES].to_numpy(dtype=float)
    y_all = encode_target(df["Diabetes_012"].to_numpy(), task=task)
    X_val = X_all[split.val]
    y_val = y_all[split.val]
    return X_val, y_val


def _build_grid(cfg_grid: dict | None) -> dict | None:
    """Chuyển cấu hình grid [start, stop, num] sang np.geomspace array.

    Args:
        cfg_grid: Dict với khoá s_pre và s_dia, mỗi khoá là [start, stop, num].

    Returns:
        Dict với np.ndarray hoặc None nếu cfg_grid là None.
    """
    if cfg_grid is None:
        return None
    result: dict[str, np.ndarray] = {}
    for key, spec in cfg_grid.items():
        if isinstance(spec, list) and len(spec) == 3:
            result[key] = np.geomspace(spec[0], spec[1], int(spec[2]))
        else:
            result[key] = np.asarray(spec, dtype=float)
    return result


def _process_run(
    run_name: str,
    model_key: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    no_log: bool,
) -> dict:
    """Xử lý một run: hiệu chuẩn, chọn ngưỡng, lưu artifact.

    Args:
        run_name: Tên run đầy đủ (vd. 'logreg_multiclass_full').
        model_key: Khóa mô hình ('majority', 'logreg', 'xgb', 'mlp').
        X_val: Ma trận đặc trưng validation (thô, chưa scale).
        y_val: Nhãn validation.
        cfg: Cấu hình từ evaluate.yaml.
        no_log: Nếu True thì không ghi nhật ký.

    Returns:
        dict: Thông tin postproc cho run này.
    """
    cal_cfg = cfg.get("calibration", {})
    thr_cfg = cfg.get("threshold", {})
    n_bins: int = int(cal_cfg.get("n_bins", 10))
    n_splits: int = int(cal_cfg.get("n_splits", 5))
    candidates: list[str] = list(cal_cfg.get("candidates", ["logit_multinomial", "isotonic_ovr"]))
    precision_floor: float = float(thr_cfg.get("precision_floor", 0.30))
    grid_cfg = thr_cfg.get("grid", None)
    grid = _build_grid(grid_cfg)

    run = load_run(run_name)
    # Scale X_val qua scaler của run
    Xs_val = run.scaler.transform(X_val)

    # --- majority: không hiệu chuẩn ---
    if model_key == "majority":
        postproc: dict = {
            "run": run_name,
            "task": run.meta.get("task", ""),
            "variant": run.meta.get("variant", ""),
            "calibration": {},
            "thresholds": {
                "scale": [1.0, 1.0, 1.0],
                "val_macro_f1_argmax": None,
                "val_macro_f1_scaled": None,
                "val_recall": None,
                "val_precision": None,
            },
            "note": "majority không hiệu chuẩn; hệ số (1,1,1)",
        }
        _write_postproc(run_name, postproc)
        print(f"[{run_name}] majority → bỏ qua hiệu chuẩn")
        return postproc

    # --- Dự đoán xác suất gốc trên validation ---
    p_val = run.predict_proba_scaled(Xs_val, calibrated=False)

    # Đảm bảo xác suất hợp lệ
    assert p_val.shape[0] == len(y_val), "Số mẫu val không khớp"
    assert np.abs(p_val.sum(axis=1) - 1.0).max() < 1e-6, "Xác suất không tổng 1"
    assert np.all(p_val >= 0) and np.all(p_val <= 1 + 1e-9), "Xác suất ngoài [0,1]"

    # ECE và Brier trước hiệu chuẩn
    ece_before_dict = ece_ovr(y_val, p_val, n_bins=n_bins)
    ece_before = float(ece_before_dict["macro"])
    ece_per_class_before = [float(x) for x in ece_before_dict["per_class"]]
    brier_before = float(brier_multiclass(y_val, p_val))

    print(f"[{run_name}] ECE trước hiệu chuẩn: {ece_before:.4f} | Brier: {brier_before:.4f}")

    # --- Chọn phương pháp hiệu chuẩn tốt nhất (OOF) ---
    # Thử candidates; nếu n_splits quá lớn cho lớp thiểu số, giảm xuống 3
    sel_result = None
    n_splits_used = n_splits
    for try_splits in [n_splits, 3]:
        try:
            sel_result = select_calibration_method(
                p_val, y_val,
                candidates=tuple(candidates),
                n_bins=n_bins,
                n_splits=try_splits,
                seed=42,
            )
            n_splits_used = try_splits
            break
        except Exception as e:
            print(f"[{run_name}] select_calibration_method n_splits={try_splits} lỗi: {e}")
            if try_splits == 3:
                raise

    chosen_method: str = sel_result["chosen"]  # type: ignore[index]
    print(f"[{run_name}] Phương pháp được chọn: {chosen_method}")

    # --- OOF calibrated với phương pháp được chọn ---
    oof_proba = oof_calibrated(
        chosen_method, p_val, y_val, n_splits=n_splits_used, seed=42
    )

    # Kiểm tra ràng buộc OOF
    assert oof_proba.shape == p_val.shape, "OOF shape không khớp"
    assert np.abs(oof_proba.sum(axis=1) - 1.0).max() < 1e-6, "OOF xác suất không tổng 1"
    assert np.all(oof_proba >= 0) and np.all(oof_proba <= 1 + 1e-9), "OOF ngoài [0,1]"

    # ECE và Brier sau hiệu chuẩn (OOF)
    ece_after_dict = ece_ovr(y_val, oof_proba, n_bins=n_bins)
    ece_after_oof = float(ece_after_dict["macro"])
    ece_per_class_after = [float(x) for x in ece_after_dict["per_class"]]
    brier_after_oof = float(brier_multiclass(y_val, oof_proba))

    print(
        f"[{run_name}] ECE sau OOF: {ece_after_oof:.4f} | Brier OOF: {brier_after_oof:.4f}"
    )

    # --- Fit calibrator trên toàn bộ validation ---
    cal = fit_calibrator(chosen_method, p_val, y_val)
    cal_path = MODELS_DIR / run_name / "calibrator.joblib"
    save_calibrator(cal, cal_path)
    print(f"[{run_name}] Đã lưu calibrator → {cal_path}")

    # --- Chọn hệ số ngưỡng (trên OOF calibrated) ---
    n_classes = p_val.shape[1]
    task = run.meta.get("task", "multiclass")

    if n_classes > 2 or task == "multiclass":
        thr_result = select_scale_factors(y_val, oof_proba, objective="macro_f1", grid=grid)
        thr_info: dict = {
            "scale": thr_result["scale"],
            "val_macro_f1_argmax": thr_result["val_macro_f1_argmax"],
            "val_macro_f1_scaled": thr_result["val_macro_f1_scaled"],
            "val_recall": thr_result["val_recall"],
            "val_precision": thr_result["val_precision"],
        }
    else:
        thr_result = select_binary_threshold(
            y_val, oof_proba[:, 1], objective="macro_f1",
            precision_floor=precision_floor,
        )
        thr_info = {
            "threshold": thr_result["threshold"],
            "val_macro_f1_argmax": None,
            "val_macro_f1_scaled": thr_result.get("val_macro_f1"),
            "val_recall": thr_result["val_recall"],
            "val_precision": thr_result["val_precision"],
        }

    # Kiểm tra: val_macro_f1_scaled >= val_macro_f1_argmax
    if thr_info.get("val_macro_f1_argmax") is not None and thr_info.get("val_macro_f1_scaled") is not None:
        assert thr_info["val_macro_f1_scaled"] >= thr_info["val_macro_f1_argmax"] - 1e-9, (
            f"[{run_name}] val_macro_f1_scaled ({thr_info['val_macro_f1_scaled']:.4f}) "
            f"< val_macro_f1_argmax ({thr_info['val_macro_f1_argmax']:.4f})"
        )

    print(
        f"[{run_name}] F1 argmax={thr_info.get('val_macro_f1_argmax', '—'):.4f} "
        f"→ scaled={thr_info.get('val_macro_f1_scaled', '—'):.4f} | "
        f"scale={thr_info.get('scale', '—')}"
    )

    # --- Ghi postproc JSON ---
    postproc = {
        "run": run_name,
        "task": task,
        "variant": run.meta.get("variant", ""),
        "calibration": {
            "method": chosen_method,
            "candidates": candidates,
            "ece_macro_before": ece_before,
            "ece_macro_after_oof": ece_after_oof,
            "brier_before": brier_before,
            "brier_after_oof": brier_after_oof,
            "ece_per_class_before": ece_per_class_before,
            "ece_per_class_after_oof": ece_per_class_after,
        },
        "thresholds": thr_info,
        "note": (
            "hiệu chuẩn và ngưỡng chọn trên cùng validation; "
            "dùng out-of-fold để giảm lạc quan"
        ),
    }

    _write_postproc(run_name, postproc)
    return postproc


def _write_postproc(run_name: str, postproc: dict) -> None:
    """Ghi file postproc JSON.

    Args:
        run_name: Tên run.
        postproc: Nội dung cần ghi.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / f"postproc_{run_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(postproc, f, ensure_ascii=False, indent=2)
    print(f"[{run_name}] Đã ghi → {out_path}")


def _select_best_model(
    runs: list[str],
    model_keys: list[str],
    task: str,
    variant: str,
    simplicity_margin: float,
    prefer_order: list[str],
) -> None:
    """Chọn mô hình tốt nhất theo quy tắc A6 và ghi model_selection.json.

    Quy tắc A6:
      - Xếp hạng logreg, xgb, mlp theo AUC OvR macro validation (xác suất gốc).
      - Nếu mô hình AUC cao nhất hơn logreg < simplicity_margin → chọn logreg.
      - Thứ tự ưu tiên khi hòa: logreg → xgb → mlp.

    Args:
        runs: Danh sách tên run đầy đủ.
        model_keys: Khoá mô hình tương ứng.
        task: Tên task.
        variant: Biến thể.
        simplicity_margin: Biên đơn giản hóa (0.003).
        prefer_order: Thứ tự ưu tiên khi hòa.
    """
    # Nạp AUC từ train_<run>.json
    candidates: dict[str, float] = {}
    for run_name, model_key in zip(runs, model_keys):
        if model_key == "majority":
            continue
        metric_path = METRICS_DIR / f"train_{run_name}.json"
        if not metric_path.is_file():
            print(f"[model_selection] Không tìm thấy {metric_path} — bỏ qua")
            continue
        with open(metric_path, encoding="utf-8") as f:
            d = json.load(f)
        auc = d.get("val", {}).get("roc_auc_ovr_macro", None)
        if auc is None:
            print(f"[model_selection] Không có val.roc_auc_ovr_macro trong {metric_path}")
            continue
        candidates[run_name] = float(auc)

    if not candidates:
        print("[model_selection] Không có ứng viên nào — bỏ qua")
        return

    # Tìm logreg trong candidates
    logreg_run = None
    logreg_auc = None
    for run_name in candidates:
        mk = run_name.split("_")[0]
        if mk == "logreg":
            logreg_run = run_name
            logreg_auc = candidates[run_name]
            break

    # Mô hình AUC cao nhất (với thứ tự ưu tiên khi hòa theo prefer_order)
    def priority(run_name: str) -> int:
        mk = run_name.split("_")[0]
        return prefer_order.index(mk) if mk in prefer_order else len(prefer_order)

    best_run = max(candidates, key=lambda r: (candidates[r], -priority(r)))
    best_auc = candidates[best_run]

    # Quy tắc A6: nếu best hơn logreg < simplicity_margin → chọn logreg
    rule_applied = ""
    if logreg_run is not None and logreg_auc is not None:
        margin = best_auc - logreg_auc
        if margin < simplicity_margin:
            # Áp dụng simplicity bias → chọn logreg
            best_run = logreg_run
            rule_applied = (
                f"best AUC hơn logreg chỉ {margin:.4f} < simplicity_margin={simplicity_margin}; "
                f"chọn logreg theo quy tắc A6"
            )
        else:
            rule_applied = (
                f"best AUC hơn logreg {margin:.4f} >= simplicity_margin={simplicity_margin}; "
                f"chọn {best_run.split('_')[0]}"
            )
    else:
        rule_applied = f"không có logreg để so sánh; chọn {best_run}"

    print(f"[model_selection] Mô hình tốt nhất: {best_run}")
    print(f"[model_selection] Quy tắc: {rule_applied}")

    # Nạp hoặc tạo model_selection.json
    sel_path = METRICS_DIR / "model_selection.json"
    if sel_path.is_file():
        with open(sel_path, encoding="utf-8") as f:
            sel_data = json.load(f)
    else:
        sel_data = {}

    group_key = f"{task}_{variant}"
    sel_data[group_key] = {
        "best": best_run,
        "candidates": {r: {"val_auc": v} for r, v in candidates.items()},
        "rule": rule_applied,
    }

    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(sel_data, f, ensure_ascii=False, indent=2)
    print(f"[model_selection] Đã ghi → {sel_path}")


def _append_log(entries: list[dict]) -> None:
    """Thêm dòng nhật ký vào docs/experiment_log.md.

    Args:
        entries: Danh sách dict với khoá date, model, config, auc, macro_f1, note.
    """
    with open(LOG_PATH, encoding="utf-8") as f:
        content = f.read()

    today = datetime.date.today().isoformat()
    new_rows = []
    for e in entries:
        new_rows.append(
            f"| {e['date']} | {e['model']} | {e['config']} | {e['auc']} | {e['macro_f1']} | {e['note']} |"
        )

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(new_rows) + "\n")

    print(f"[log] Đã thêm {len(new_rows)} dòng vào {LOG_PATH}")


def main() -> None:
    """Điểm vào chính của script."""
    parser = argparse.ArgumentParser(
        description="Hiệu chuẩn xác suất và chọn hệ số ngưỡng trên validation"
    )
    parser.add_argument("--task", default="multiclass", help="Loại bài toán (multiclass/binary)")
    parser.add_argument("--dedup", action="store_true", help="Dùng biến thể dedup thay vì full")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["majority", "logreg", "xgb", "mlp"],
        help="Danh sách model key cần xử lý",
    )
    parser.add_argument("--no-log", action="store_true", help="Không ghi nhật ký experiment")
    args = parser.parse_args()

    task: str = args.task
    variant: str = "dedup" if args.dedup else "full"
    model_keys: list[str] = args.models
    no_log: bool = args.no_log

    cfg = _load_config()
    cal_cfg = cfg.get("calibration", {})
    sel_cfg = cfg.get("selection", {})
    simplicity_margin: float = float(sel_cfg.get("simplicity_margin", 0.003))
    prefer_order: list[str] = list(sel_cfg.get("prefer_order", ["logreg", "xgb", "mlp"]))

    print(f"=== calibrate_and_threshold | task={task} | variant={variant} ===")
    print(f"models: {model_keys}")

    # Nạp dữ liệu validation (một lần duy nhất — không nạp test)
    print("Đang nạp dữ liệu validation…")
    X_val, y_val = _load_val_data(task, variant)
    print(f"X_val: {X_val.shape}, y_val unique: {np.unique(y_val, return_counts=True)}")

    runs: list[str] = []
    log_entries: list[dict] = []
    today = datetime.date.today().isoformat()

    for mk in model_keys:
        run_name = f"{mk}_{task}_{variant}"
        runs.append(run_name)
        print(f"\n--- Xử lý: {run_name} ---")
        try:
            postproc = _process_run(run_name, mk, X_val, y_val, cfg, no_log)
        except Exception as e:
            print(f"[LỖI] {run_name}: {e}")
            raise

        # Chuẩn bị dòng log
        if mk != "majority" and postproc.get("calibration"):
            cal_info = postproc["calibration"]
            thr_info = postproc.get("thresholds", {})
            ece_before = cal_info.get("ece_macro_before", "—")
            ece_after = cal_info.get("ece_macro_after_oof", "—")
            f1_scaled = thr_info.get("val_macro_f1_scaled", "—")
            # Nạp AUC từ train metrics
            metric_path = METRICS_DIR / f"train_{run_name}.json"
            auc_val = "—"
            if metric_path.is_file():
                with open(metric_path, encoding="utf-8") as f:
                    d = json.load(f)
                auc_val = f"{d.get('val', {}).get('roc_auc_ovr_macro', '—'):.3f}"
            note = (
                f"hiệu chuẩn, val; ECE {ece_before:.4f}→{ece_after:.4f}; "
                f"macro-F1 scaled={f1_scaled:.4f}"
                if isinstance(ece_before, float) and isinstance(ece_after, float) and isinstance(f1_scaled, float)
                else "hiệu chuẩn, val"
            )
            log_entries.append({
                "date": today,
                "model": mk,
                "config": "configs/evaluate.yaml",
                "auc": auc_val,
                "macro_f1": f"{f1_scaled:.3f}" if isinstance(f1_scaled, float) else "—",
                "note": note,
            })

    # --- Chọn mô hình tốt nhất ---
    print("\n=== Chọn mô hình tốt nhất ===")
    _select_best_model(runs, model_keys, task, variant, simplicity_margin, prefer_order)

    # --- Ghi nhật ký ---
    if not no_log and log_entries:
        _append_log(log_entries)

    # --- Kiểm tra tiêu chí ---
    print("\n=== Kiểm tra tiêu chí ===")
    all_pass = True
    for mk in [k for k in model_keys if k != "majority"]:
        run_name = f"{mk}_{task}_{variant}"
        postproc_path = METRICS_DIR / f"postproc_{run_name}.json"
        if not postproc_path.is_file():
            print(f"  [{run_name}] FAIL — không tìm thấy postproc file")
            all_pass = False
            continue
        with open(postproc_path, encoding="utf-8") as f:
            pp = json.load(f)
        cal = pp.get("calibration", {})
        ece_before = cal.get("ece_macro_before", float("inf"))
        ece_after = cal.get("ece_macro_after_oof", float("inf"))
        thr = pp.get("thresholds", {})
        f1_argmax = thr.get("val_macro_f1_argmax")
        f1_scaled = thr.get("val_macro_f1_scaled")

        ok_ece_reduce = ece_after < ece_before
        ok_ece_threshold = ece_after <= 0.03
        ok_f1 = (f1_argmax is None or f1_scaled is None or f1_scaled >= f1_argmax - 1e-9)

        status = "PASS" if (ok_ece_reduce and ok_ece_threshold and ok_f1) else "FAIL"
        if not (ok_ece_reduce and ok_ece_threshold and ok_f1):
            all_pass = False

        f1_argmax_str = f"{f1_argmax:.4f}" if isinstance(f1_argmax, float) else "—"
        f1_scaled_str = f"{f1_scaled:.4f}" if isinstance(f1_scaled, float) else "—"
        print(
            f"  [{run_name}] {status} | "
            f"ECE {ece_before:.4f}→{ece_after:.4f} "
            f"(reduce={'✓' if ok_ece_reduce else '✗'}, ≤0.03={'✓' if ok_ece_threshold else '✗'}) | "
            f"F1 argmax={f1_argmax_str} "
            f"scaled={f1_scaled_str} "
            f"(≥argmax={'✓' if ok_f1 else '✗'})"
        )

    if not all_pass:
        print("\n[CẢNH BÁO] Một hoặc nhiều run không đạt tiêu chí ECE/F1 theo plan.")
        print("Dừng và báo cáo — không nới tiêu chí.")
        sys.exit(1)
    else:
        print("\n✓ Tất cả tiêu chí đã đạt.")


if __name__ == "__main__":
    main()
