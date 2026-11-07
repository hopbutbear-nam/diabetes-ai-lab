"""Đọc configs/safety_rules.yaml, đánh giá luật an toàn dựa trên ngữ cảnh người dùng.

Luật an toàn cố định luôn được ưu tiên hơn mô hình học máy.
Không dùng eval trên chuỗi tùy ý; chỉ dùng AST với tập nút an toàn đã được kiểm tra nghiêm ngặt.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from diabetes_ai.paths import CONFIGS_DIR

_SAFE_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
)

LEVEL_PRIORITY: dict[str, int] = {
    "danger": 0,
    "warn": 1,
    "info": 2,
}

VALID_LEVELS: set[str] = set(LEVEL_PRIORITY.keys())


@dataclass(frozen=True)
class Rule:
    """Định nghĩa một luật an toàn."""

    id: str
    level: str
    when: str
    text: str


@dataclass
class Alert:
    """Cảnh báo an toàn được kích hoạt khi thỏa mãn luật."""

    rule_id: str
    level: str
    text: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _validate_safe_ast(expr: str) -> ast.Expression:
    """Kiểm tra cú pháp biểu thức và đảm bảo chỉ chứa các nút AST an toàn."""
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as exc:
        raise ValueError(f"Biểu thức luật không an toàn hoặc lỗi cú pháp: '{expr}'") from exc

    if any(not isinstance(n, _SAFE_NODES) for n in ast.walk(tree)):
        raise ValueError(f"Biểu thức luật không an toàn: {expr}")
    return tree


def load_rules(path: Path | str | None = None) -> list[Rule]:
    """Nạp danh sách luật an toàn từ file YAML.

    Args:
        path: Đường dẫn tới file YAML cấu hình luật an toàn. Nếu None, dùng mặc định
            configs/safety_rules.yaml.

    Returns:
        list[Rule]: Danh sách các luật đã được kiểm tra an toàn.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu ID luật bị trùng, mức cảnh báo không hợp lệ hoặc biểu thức không an toàn.
    """
    rule_path = Path(path) if path is not None else (CONFIGS_DIR / "safety_rules.yaml")
    if not rule_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file luật an toàn: {rule_path}")

    with open(rule_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_rules = (data or {}).get("rules", [])
    seen_ids: set[str] = set()
    rules: list[Rule] = []

    for item in raw_rules:
        rule_id = str(item.get("id", "")).strip()
        if not rule_id:
            raise ValueError("Luật thiếu trường 'id'")
        if rule_id in seen_ids:
            raise ValueError(f"ID luật bị trùng: {rule_id}")
        seen_ids.add(rule_id)

        level = str(item.get("level", "")).strip()
        if level not in VALID_LEVELS:
            raise ValueError(
                f"Mức cảnh báo không hợp lệ: '{level}' cho luật '{rule_id}' (chỉ nhận: {sorted(VALID_LEVELS)})"
            )

        when = str(item.get("when", "")).strip()
        if not when:
            raise ValueError(f"Luật '{rule_id}' thiếu biểu thức điều kiện 'when'")
        _validate_safe_ast(when)

        text = str(item.get("text", "")).strip()
        rules.append(Rule(id=rule_id, level=level, when=when, text=text))

    return rules


def evaluate(context: dict[str, Any], rules: list[Rule] | None = None) -> list[Alert]:
    """Đánh giá danh sách luật an toàn trong một ngữ cảnh dữ liệu người dùng.

    Args:
        context: Từ điển chứa các biến ngữ cảnh người dùng. Giá trị None khiến
            luật liên quan không chạy vì thiếu dữ liệu đầu vào.
        rules: Danh sách luật cần đánh giá. Nếu None, tự động nạp từ load_rules().

    Returns:
        list[Alert]: Danh sách cảnh báo được kích hoạt, sắp xếp theo mức độ ưu tiên
            danger -> warn -> info.
    """
    if rules is None:
        rules = load_rules()

    alerts: list[Alert] = []

    for rule in rules:
        tree = _validate_safe_ast(rule.when)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

        # Nếu biến cần thiết bị thiếu hoặc mang giá trị None -> luật không chạy
        if any(context.get(n) is None for n in names):
            continue

        compiled = compile(tree, "<rule>", "eval")
        # Chạy trong sandbox không có __builtins__
        ok = bool(eval(compiled, {"__builtins__": {}}, context))  # nosec
        if ok:
            evidence = {n: context[n] for n in names if n in context}
            alerts.append(
                Alert(
                    rule_id=rule.id,
                    level=rule.level,
                    text=rule.text,
                    evidence=evidence,
                )
            )

    # Sắp xếp cảnh báo: danger -> warn -> info
    alerts.sort(key=lambda a: LEVEL_PRIORITY.get(a.level, 99))
    return alerts


def top_level(alerts: list[Alert]) -> str | None:
    """Trả về mức cảnh báo nghiêm trọng nhất trong danh sách cảnh báo.

    Args:
        alerts: Danh sách Alert.

    Returns:
        str | None: 'danger', 'warn', 'info' hoặc None nếu không có cảnh báo.
    """
    if not alerts:
        return None
    for lvl in ("danger", "warn", "info"):
        if any(a.level == lvl for a in alerts):
            return lvl
    return None
