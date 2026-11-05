"""Báo cáo số liệu đánh giá, thống kê cảnh báo rủi ro và quản lý khóa tập test.

Đảm bảo nguyên tắc bảo toàn tập test: chỉ đánh giá một lần duy nhất trừ khi có cờ ép buộc.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(obj: Any, path: str | Path) -> Path:
    """Ghi đối tượng Python sang file JSON UTF-8 với định dạng đẹp (indent=2).

    Tự động chuyển đổi các kiểu dữ liệu NumPy sang kiểu thuần Python.
    Tạo thư mục cha nếu chưa tồn tại.

    Args:
        obj: Đối tượng cần ghi.
        path: Đường dẫn file JSON đầu ra.

    Returns:
        Path: Đường dẫn file JSON đã ghi.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, dict):
            return {str(k): _convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [_convert(v) for v in o]
        return o

    converted = _convert(obj)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    return out_path


def file_sha256(path: str | Path) -> str:
    """Tính chuỗi băm SHA-256 của file theo từng khối 64KB.

    Args:
        path: Đường dẫn tới file cần tính hash.

    Returns:
        str: Chuỗi mã băm SHA-256 hexdigest.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_lock(path: str | Path) -> dict[str, Any]:
    """Đọc file khóa test. Nếu file chưa tồn tại hoặc rỗng, trả về cấu trúc rỗng.

    Args:
        path: Đường dẫn file khóa test JSON.

    Returns:
        dict: Dữ liệu khóa chứa danh sách các lần đánh giá ("evaluations").
    """
    lock_path = Path(path)
    if not lock_path.exists():
        return {"evaluations": []}

    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "evaluations" in data and isinstance(data["evaluations"], list):
            return data
    except Exception:
        pass
    return {"evaluations": []}


def acquire_test_lock(
    path: str | Path,
    runs: list[str],
    model_hashes: dict[str, str],
    force: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Khóa tập test để đảm bảo tập test chỉ được đánh giá duy nhất một lần.

    Nếu đã có lần đánh giá trước đó và force=False, hàm sẽ ném ra ngoại lệ RuntimeError.
    Nếu thành công, hàm sẽ ghi nhận thêm mục đánh giá mới vào file JSON và trả về nội dung khóa.

    Args:
        path: Đường dẫn file test_lock.json.
        runs: Danh sách tên run được đánh giá.
        model_hashes: Từ điển mapping tên artifact/model sang chuỗi sha256.
        force: Cờ ép buộc đánh giá lại (--force-reeval).
        reason: Lý do ép buộc đánh giá lại (--reason).

    Returns:
        dict: Nội dung khóa đã cập nhật.
    """
    lock_path = Path(path)
    lock_data = read_lock(lock_path)
    evaluations = lock_data.get("evaluations", [])

    if len(evaluations) > 0 and not force:
        last_eval = evaluations[-1]
        last_utc = last_eval.get("utc", "không rõ ngày")
        raise RuntimeError(
            f"Test đã được đánh giá ngày {last_utc}; dùng --force-reeval --reason ..."
        )

    n = len(evaluations) + 1
    utc_str = datetime.now(timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "n": n,
        "utc": utc_str,
        "runs": list(runs),
        "model_hashes": dict(model_hashes),
        "forced": bool(force),
        "reason": reason,
    }
    evaluations.append(entry)
    lock_data["evaluations"] = evaluations
    write_json(lock_data, lock_path)
    return lock_data


def risk_alert_stats(
    y_true: np.ndarray | list[int],
    risk_index: np.ndarray | list[float],
    thresholds: dict[str, float] | None = None,
    positive_class: int = 2,
) -> dict[str, Any]:
    """Tính các số liệu thống kê cảnh báo rủi ro theo từng ngưỡng.

    Với mỗi ngưỡng xác định:
      - alert_rate: Tỉ lệ mẫu được cảnh báo (risk_index >= ngưỡng).
      - ppv: Giá trị tiên đoán dương tính (Precision: TP / Số cảnh báo).
      - recall: Độ nhạy (TP / Số mẫu dương thực tế).
      - false_positive_rate: Tỉ lệ dương tính giả (FP / Số mẫu âm).
      - lift: Tỉ số tăng cường (PPV / Tỉ lệ dương nền).

    Args:
        y_true: Nhãn thực tế (1D).
        risk_index: Chỉ số nguy cơ dự đoán (xác suất thuộc [0, 1]).
        thresholds: Từ điển các ngưỡng (ví dụ: {"medium_or_above": 0.10, "high": 0.25}).
        positive_class: Nhãn của lớp dương (mặc định 2: tiểu đường).

    Returns:
        dict: Từ điển chứa tỉ lệ dương nền ("base_rate") và số liệu theo từng mức ngưỡng.
    """
    y_arr = np.asarray(y_true)
    r_arr = np.asarray(risk_index, dtype=float)

    if len(y_arr) != len(r_arr):
        raise ValueError(
            f"Kích thước y_true ({len(y_arr)}) và risk_index ({len(r_arr)}) không khớp."
        )

    n = len(y_arr)
    pos_mask = y_arr == positive_class
    n_pos = int(np.sum(pos_mask))
    n_neg = n - n_pos
    base_rate = float(n_pos / n) if n > 0 else 0.0

    if thresholds is None:
        thresholds = {"medium_or_above": 0.10, "high": 0.25}

    results: dict[str, Any] = {
        "n": n,
        "n_pos": n_pos,
        "base_rate": base_rate,
    }

    for name, th in thresholds.items():
        th_val = float(th)
        alert_mask = r_arr >= th_val
        n_alert = int(np.sum(alert_mask))
        alert_rate = float(n_alert / n) if n > 0 else 0.0

        tp = int(np.sum(alert_mask & pos_mask))
        fp = int(np.sum(alert_mask & ~pos_mask))
        fn = int(np.sum(~alert_mask & pos_mask))
        tn = int(np.sum(~alert_mask & ~pos_mask))

        ppv = float(tp / n_alert) if n_alert > 0 else 0.0
        recall = float(tp / n_pos) if n_pos > 0 else 0.0
        fpr = float(fp / n_neg) if n_neg > 0 else 0.0
        lift = float(ppv / base_rate) if base_rate > 0 else 0.0

        stat = {
            "threshold": th_val,
            "alert_rate": alert_rate,
            "ppv": ppv,
            "recall": recall,
            "false_positive_rate": fpr,
            "lift": lift,
            "n_alert": n_alert,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        results[name] = stat

    # Thêm alias tiếng Việt / chuẩn hóa để thuận tiện tra cứu
    if "low" in thresholds and "medium_or_above" not in results:
        results["medium_or_above"] = results["low"]
        results["moderate_or_above"] = results["low"]
    if "medium" in thresholds and "high" not in results:
        results["high"] = results["medium"]
        results["cao"] = results["medium"]
    if "high" in results and "cao" not in results:
        results["cao"] = results["high"]
    if "medium_or_above" in results and "trung_binh_tro_len" not in results:
        results["trung_binh_tro_len"] = results["medium_or_above"]
    elif "moderate_or_above" in results and "trung_binh_tro_len" not in results:
        results["trung_binh_tro_len"] = results["moderate_or_above"]

    return results
