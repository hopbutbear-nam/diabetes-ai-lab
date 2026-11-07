"""Kiểm thử tính đầy đủ của tài liệu và miễn trừ trách nhiệm y tế."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_has_disclaimer() -> None:
    """README phải có phần cảnh báo/miễn trừ trách nhiệm y tế."""
    readme_path = ROOT / "README.md"
    assert readme_path.exists(), "README.md không tồn tại"
    text = readme_path.read_text(encoding="utf-8").lower()
    assert (
        "miễn trừ trách nhiệm" in text
        or "không phải chẩn đoán" in text
        or "chỉ để học tập" in text
    ), "README.md thiếu cảnh báo miễn trừ trách nhiệm y tế"


def test_model_card_sections_and_disclaimer() -> None:
    """Model card phải có đủ các mục bắt buộc và miễn trừ trách nhiệm."""
    mc_path = ROOT / "reports" / "model_card.md"
    assert mc_path.exists(), "reports/model_card.md không tồn tại"
    text = mc_path.read_text(encoding="utf-8")

    required_sections = [
        "Mục đích và phạm vi",
        "Dữ liệu",
        "Mô hình và huấn luyện",
        "Hiệu năng",
        "Hiệu chuẩn",
        "Dòng trùng lặp và đối chứng",
        "Mức nguy cơ và cảnh báo",
        "Nhóm bị bỏ sót",
        "Cách dùng đúng",
        "Cách dùng sai (không dùng để chẩn đoán)",
        "Hạn chế",
    ]
    for sec in required_sections:
        assert sec in text, f"reports/model_card.md thiếu mục: {sec}"

    assert (
        "không phải chẩn đoán" in text.lower()
    ), "reports/model_card.md thiếu dòng 'không phải chẩn đoán'"


def test_reports_have_no_unfilled_auto_markers() -> None:
    """Các báo cáo trong reports/ không còn khối AUTO rỗng."""
    report_files = [
        ROOT / "reports" / "model_card.md",
        ROOT / "reports" / "fairness_report.md",
        ROOT / "reports" / "final_report.md",
    ]
    empty_pat = re.compile(r"<!--\s*AUTO:(.*?)\s*-->\s*<!--\s*/AUTO\s*-->", re.DOTALL)

    for rp in report_files:
        assert rp.exists(), f"File báo cáo không tồn tại: {rp}"
        content = rp.read_text(encoding="utf-8")
        assert "<!-- AUTO:" in content, f"{rp.name} không chứa khối AUTO nào"
        assert not empty_pat.search(content), f"{rp.name} còn khối AUTO rỗng chưa được điền số liệu"


def test_safety_doc_mentions_confirmatory_test() -> None:
    """Tài liệu an toàn và đạo đức phải nêu rõ xét nghiệm xác nhận lâm sàng."""
    safety_path = ROOT / "docs" / "safety_and_ethics.md"
    assert safety_path.exists(), "docs/safety_and_ethics.md không tồn tại"
    text = safety_path.read_text(encoding="utf-8")
    assert (
        "xét nghiệm xác nhận" in text.lower()
    ), "docs/safety_and_ethics.md phải chứa cụm từ 'xét nghiệm xác nhận'"
