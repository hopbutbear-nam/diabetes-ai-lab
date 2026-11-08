"""Sinh docs/data_dictionary.md từ số liệu đã đo (không gõ tay số liệu)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from diabetes_ai.data.schema import CLASSES, COUNT_FEATURES, FEATURE_RANGES, FEATURES
from diabetes_ai.features.feature_names import (
    AGE_LABELS,
    EDUCATION_LABELS,
    GENHLTH_LABELS,
    INCOME_LABELS,
    SEX_LABELS,
    VI_FEATURE,
)
from diabetes_ai.paths import DOCS_DIR, METRICS_DIR, RAW_DIR, SPLITS_DIR

CODE_TABLES = {
    "Age": AGE_LABELS,
    "Education": EDUCATION_LABELS,
    "GenHlth": GENHLTH_LABELS,
    "Income": INCOME_LABELS,
    "Sex": SEX_LABELS,
}


def vn(n: int | float) -> str:
    """Định dạng số nguyên theo kiểu Việt Nam (dấu chấm ngăn nghìn)."""
    return f"{int(n):,}".replace(",", ".")


def pct(x: float) -> str:
    """Định dạng phần trăm với dấu phẩy thập phân."""
    return f"{100 * x:.1f}".replace(".", ",") + "%"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _kind(name: str, lo: float, hi: float) -> str:
    if (lo, hi) == (0.0, 1.0):
        return "nhị phân"
    if name in COUNT_FEATURES:
        return "đếm (số ngày)"
    if name == "BMI":
        return "liên tục"
    return "thứ bậc"


def _coding(name: str, lo: float, hi: float) -> str:
    table = CODE_TABLES.get(name)
    if table:
        return "; ".join(f"{k} = {v}" for k, v in table.items())
    return "0 = không, 1 = có" if (lo, hi) == (0.0, 1.0) else f"{lo:g} – {hi:g}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    val = _read(METRICS_DIR / "data_validation.json")
    csv_info = _read(METRICS_DIR / "env_check.json")["csv"]
    full = _read(SPLITS_DIR / "full_seed42" / "manifest.json")
    dedup = _read(SPLITS_DIR / "dedup_seed42" / "manifest.json")
    csv_path = next(RAW_DIR.glob("*.csv"))
    downloaded = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d")

    L: list[str] = []
    L += ["# Từ điển dữ liệu — BRFSS 2015 (Diabetes Health Indicators)", ""]
    L += ["> Tệp này do `scripts/make_data_dictionary.py` sinh từ số liệu đo; không sửa tay các con số.", ""]
    L += ["## 1. Nguồn", ""]
    L += [f"- Tệp: `{csv_path.name}` (bộ \"Diabetes Health Indicators\" từ khảo sát BRFSS 2015 của CDC, bản đăng trên Kaggle).",
          f"- Ngày tệp (theo `stat`): {downloaded}; kích thước {vn(csv_info['bytes'])} byte; sha256 `{csv_info['sha256']}`.",
          f"- Số dòng: {vn(val['n_rows'])}; số cột: {val['n_cols']} (21 đặc trưng + 1 nhãn).", ""]
    L += ["## 2. Các cột", ""]
    L += ["| Cột | Tiếng Việt | Kiểu | Miền | Mã hóa |", "|---|---|---|---|---|"]
    for name in FEATURES:
        lo, hi = FEATURE_RANGES[name]
        L.append(f"| `{name}` | {VI_FEATURE[name]} | {_kind(name, lo, hi)} | {lo:g} – {hi:g} | {_coding(name, lo, hi)} |")
    L += ["", "**Lưu ý (giả định A15):** cách mã hóa Age/Income/Education/GenHlth/Sex theo mô tả của bộ dữ liệu Kaggle; "
          "**chưa đối chiếu** với codebook CDC gốc.", ""]
    L += ["## 3. Nhãn", ""]
    L += [f"- Nhãn 3 lớp `Diabetes_012`: " + "; ".join(f"{i} = {c}" for i, c in enumerate(CLASSES)) + ".",
          "- Nhãn nhị phân: 0 = không; 1 = tiền tiểu đường hoặc tiểu đường.",
          "- Số lượng toàn bộ: " + "; ".join(f"{CLASSES[int(k)]} = {vn(v)}" for k, v in val["label_counts"].items()) + ".", ""]
    L += ["## 4. Chất lượng dữ liệu", ""]
    L += [f"- Dòng trùng hoàn toàn (cả nhãn): {vn(val['n_duplicates'])} ({pct(val['n_duplicates'] / val['n_rows'])}); "
          f"còn {vn(val['n_unique_rows'])} dòng duy nhất.",
          f"- Kết luận kiểm tra tự động: `ok = {val['ok']}`.", ""]
    L += ["## 5. Quyết định về dòng trùng lặp", ""]
    L += [f"- Biến thể **`full`** (giữ dòng trùng) là biến thể **chính**: các dòng trùng là những người trả lời khác nhau "
          f"có hồ sơ giống nhau; khử trùng làm lệch tỉ lệ tiểu đường nền từ {pct(full['class_pct']['train']['2'])} "
          f"sang {pct(dedup['class_pct']['train']['2'])} và làm xác suất hiệu chuẩn sai.",
          "- Biến thể **`dedup`** (xóa bản sao thừa, giữ lần xuất hiện đầu tiên) chỉ là **đối chứng**.", ""]
    L += ["## 6. Chia tập (seed 42, phân tầng theo nhãn 3 lớp)", ""]
    L += ["| Biến thể | Train | Validation | Test |", "|---|---|---|---|"]
    for name, m in (("full", full), ("dedup", dedup)):
        s = m["sizes"]
        L.append(f"| {name} | {vn(s['train'])} | {vn(s['val'])} | {vn(s['test'])} |")
    L += ["", "`val_size = 0,15` là 15% của phần còn lại sau khi tách test (`test_size = 0,2`)."]

    out = DOCS_DIR / "data_dictionary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
