"""Kiểm thử tĩnh cho giao diện web demo sàng lọc nguy cơ tiểu đường."""

import re

from diabetes_ai.config import load_config
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
        assert expected_attr in content, (
            f"Trường đặc trưng '{feat}' không có trong form ({expected_attr})."
        )


def test_app_js_calls_endpoints_and_renders_disclaimer() -> None:
    """Kiểm tra app.js gọi các endpoint /predict, /explain, xử lý disclaimer và không lưu trữ client-side."""
    js_path = ROOT / "app" / "static" / "js" / "app.js"
    assert js_path.is_file(), "Tệp app/static/js/app.js không tồn tại."
    content = js_path.read_text(encoding="utf-8")

    assert "/predict" in content
    assert "/explain" in content
    assert "disclaimer" in content
    assert "localStorage" not in content


def test_whatif_keys_sent_by_ui_are_allowed_by_server() -> None:
    """Các khóa what-if mà app.js gửi tới /explain phải nằm trong khối `what_if` của configs/features.yaml."""
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    allowed = set(load_config("features")["what_if"].keys())

    block = re.search(r"const whatIfChanges = \{(.*?)\n\s*\};", js, re.S)
    assert block, "Không tìm thấy khối `const whatIfChanges = {...}` trong app.js"
    sent = set(re.findall(r"^\s*(\w+):", block.group(1), re.M))
    assert sent, "Khối whatIfChanges không có khóa nào"
    assert sent <= allowed, f"app.js gửi khóa không được phép: {sorted(sent - allowed)}"

    for key in sent - {"BMI"}:
        assert f'id="whatif-{key}"' in html, f"Thiếu ô bật/tắt whatif-{key} trong index.html"


def test_every_whatif_toggle_has_change_listener() -> None:
    """Mọi ô bật/tắt what-if trong index.html phải được gắn sự kiện `change` trong app.js."""
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    toggles = set(re.findall(r'<input type="checkbox" id="(whatif-\w+)"', html))
    assert toggles, "Không tìm thấy ô bật/tắt what-if nào trong index.html"
    listener = re.search(r"\[([^\]]*)\]\.forEach\(\(id\)", js)
    assert listener, "Không tìm thấy danh sách id gắn sự kiện trong app.js"
    wired = set(re.findall(r'"(whatif-\w+)"', listener.group(1)))
    assert toggles <= wired, f"Ô what-if chưa gắn sự kiện: {sorted(toggles - wired)}"
    assert wired <= toggles, f"app.js gắn sự kiện cho id không tồn tại: {sorted(wired - toggles)}"


def test_app_js_calls_consult_endpoint() -> None:
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "/consult" in js


def test_app_js_has_no_dynamic_inner_html() -> None:
    """Chỉ cho phép innerHTML để xóa nội dung (gán chuỗi rỗng); không ghép HTML từ dữ liệu."""
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    for line in js.splitlines():
        if "innerHTML" in line:
            assert '= ""' in line, f"innerHTML động không được phép: {line.strip()}"
