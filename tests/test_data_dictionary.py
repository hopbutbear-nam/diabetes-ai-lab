"""Kiểm tra từ điển dữ liệu chứa đúng các con số đã đo."""

import json

from diabetes_ai.paths import DOCS_DIR, METRICS_DIR, SPLITS_DIR


def _vn(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def test_data_dictionary_has_measured_numbers() -> None:
    text = (DOCS_DIR / "data_dictionary.md").read_text(encoding="utf-8")
    val = json.loads((METRICS_DIR / "data_validation.json").read_text(encoding="utf-8"))
    sizes = json.loads((SPLITS_DIR / "full_seed42" / "manifest.json").read_text(encoding="utf-8"))["sizes"]
    for n in (val["n_rows"], val["n_duplicates"], sizes["train"], sizes["val"], sizes["test"]):
        assert _vn(n) in text, f"Thiếu số {_vn(n)} trong docs/data_dictionary.md"
