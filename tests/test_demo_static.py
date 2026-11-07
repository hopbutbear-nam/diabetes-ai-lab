"""Kiểm thử tĩnh cho giao diện web demo sàng lọc nguy cơ tiểu đường."""

from diabetes_ai.data.schema import FEATURES
from diabetes_ai.paths import ROOT


def test_index_has_disclaimer_banner() -> None:
    """Kiểm tra trang index.html có dải cảnh báo miễn trừ trách nhiệm y tế."""
    index_path = ROOT / "app" / "index.html"
    assert index_path.is_file(), "Tệp app/index.html không tồn tại."
    content = index_path.read_text(encoding="utf-8")

    assert "Chỉ để học tập" in content
    assert "không phải chẩn đoán y khoa" in content


def test_index_has_all_21_feature_inputs() -> None:
    """Kiểm tra form có đầy đủ trường nhập liệu cho cả 21 đặc trưng với name='<tên>'."""
    index_path = ROOT / "app" / "index.html"
    content = index_path.read_text(encoding="utf-8")

    for feat in FEATURES:
        expected_attr = f'name="{feat}"'
        assert expected_attr in content, f"Trường đặc trưng '{feat}' không có trong form ({expected_attr})."


def test_app_js_calls_endpoints_and_renders_disclaimer() -> None:
    """Kiểm tra app.js gọi các endpoint /predict, /explain, xử lý disclaimer và không lưu trữ client-side."""
    js_path = ROOT / "app" / "static" / "js" / "app.js"
    assert js_path.is_file(), "Tệp app/static/js/app.js không tồn tại."
    content = js_path.read_text(encoding="utf-8")

    assert "/predict" in content
    assert "/explain" in content
    assert "disclaimer" in content
    assert "localStorage" not in content
