"""Kiểm thử bảo vệ tập test (guard tests) cho bước K06d."""

from pathlib import Path
import re
import pytest

from diabetes_ai.evaluation.report import acquire_test_lock, read_lock
from diabetes_ai.paths import ROOT


def test_include_test_only_in_evaluate_script() -> None:
    hits = [
        p
        for p in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]
        if re.search(r"include_test\s*=\s*True", p.read_text(encoding="utf-8"))
    ]
    assert [p.name for p in hits] == ["evaluate.py"]


def test_second_lock_acquisition_refused(tmp_path: Path) -> None:
    lock_file = tmp_path / "test_lock.json"
    runs = ["run1"]
    hashes = {"model": "dummy"}
    acquire_test_lock(lock_file, runs=runs, model_hashes=hashes)
    with pytest.raises(RuntimeError):
        acquire_test_lock(lock_file, runs=runs, model_hashes=hashes)


def test_force_reeval_records_reason(tmp_path: Path) -> None:
    lock_file = tmp_path / "test_lock.json"
    runs = ["run1"]
    hashes = {"model": "dummy"}
    acquire_test_lock(lock_file, runs=runs, model_hashes=hashes)
    acquire_test_lock(
        lock_file,
        runs=runs,
        model_hashes=hashes,
        force=True,
        reason="Đánh giá lại để kiểm tra",
    )
    data = read_lock(lock_file)
    assert len(data["evaluations"]) == 2
    assert data["evaluations"][1]["forced"] is True
    assert data["evaluations"][1]["reason"] == "Đánh giá lại để kiểm tra"
