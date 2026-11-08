"""Tài liệu phải dùng đúng số đo, không còn số cũ/sai."""

import json

from diabetes_ai.paths import METRICS_DIR, ROOT


def _vn(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_docs_do_not_contain_stale_numbers() -> None:
    for rel in ("README.md", "docs/architecture.md"):
        text = _read(rel)
        assert "24.206" not in text, f"{rel} còn số trùng lặp cũ 24.206"
        assert "229.474" not in text, f"{rel} còn số 229.474"
    assert "60/20/20" not in _read("README.md")
    assert "/explain/global" not in _read("docs/architecture.md")
    assert "BatchNorm" not in _read("docs/architecture.md")


def test_docs_use_measured_duplicate_counts() -> None:
    val = json.loads((METRICS_DIR / "data_validation.json").read_text(encoding="utf-8"))
    arch = _read("docs/architecture.md")
    assert _vn(val["n_duplicates"]) in arch
    assert _vn(val["n_unique_rows"]) in arch
    assert _vn(val["n_duplicates"]) in _read("README.md")
