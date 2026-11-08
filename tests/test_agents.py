"""Kiểm thử hệ đa tác tử: điều phối, luồng sự kiện và kiểm định phân tầng an toàn."""

import ast
import json
from pathlib import Path

import pytest

from diabetes_ai.agents.orchestrator import Orchestrator
from diabetes_ai.agents.safety_agent import SafetyAgent
from diabetes_ai.paths import ROOT


@pytest.fixture(autouse=True)
def _use_tiny_bundle(monkeypatch: pytest.MonkeyPatch, tiny_bundle) -> None:
    """Ép Orchestrator/RiskAgent dùng bundle nhỏ của fixture, không phụ thuộc models/best_model.joblib."""
    monkeypatch.setenv("MODEL_PATH", str(tiny_bundle))


def _load_sample_profile() -> dict:
    profile_path = ROOT / "tests" / "fixtures" / "sample_profile.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_orchestrator_event_order_ends_with_final():
    """Kiểm tra chuỗi sự kiện tuân thủ giao thức: bắt đầu bằng handoff/tool và kết thúc bằng final."""
    profile = _load_sample_profile()
    orchestrator = Orchestrator()
    result = orchestrator.consult(profile)

    events = result["events"]
    assert len(events) >= 6

    # Sự kiện cuối cùng bắt buộc là 'final'
    assert events[-1]["event"] == "final"
    assert events[-1]["type"] == "final"

    # Kiểm tra tập các sự kiện sinh ra nằm trong bộ giao thức chuẩn
    valid_event_types = {"speak", "handoff", "tool", "cite", "alert", "final"}
    event_sequence = [e["event"] for e in events]
    for ev in event_sequence:
        assert ev in valid_event_types

    # Kiểm tra thứ tự tuần tự của luồng: risk_agent -> safety_agent -> knowledge_agent -> final
    agents_in_order = []
    for e in events:
        agent_name = e.get("agent") or e.get("to")
        if agent_name and (not agents_in_order or agents_in_order[-1] != agent_name):
            agents_in_order.append(agent_name)

    assert "risk_agent" in agents_in_order
    assert "safety_agent" in agents_in_order
    assert "knowledge_agent" in agents_in_order

    idx_risk = agents_in_order.index("risk_agent")
    idx_safety = agents_in_order.index("safety_agent")
    idx_knowledge = agents_in_order.index("knowledge_agent")
    assert idx_risk < idx_safety < idx_knowledge


def test_danger_alert_raises_level():
    """Kiểm tra khi luật an toàn báo nguy cơ mức 'danger' (R1: glucose >= 250), mức cuối cùng luôn là 'cao'."""
    profile = _load_sample_profile()
    # Gán đường huyết glucose rất cao kích hoạt luật nguy hiểm R1 (level: danger)
    profile["glucose"] = 280.0

    orchestrator = Orchestrator()
    result = orchestrator.consult(profile)

    # Đảm bảo cảnh báo mức danger xuất hiện
    assert any(a["level"] == "danger" for a in result["alerts"])

    # Phân tầng nguy cơ cuối cùng bắt buộc phải được nâng lên 'cao'
    assert result["final_level"] == "cao"
    assert result["risk_level"] == "cao"
    assert result["events"][-1]["risk_level"] == "cao"


def test_final_has_disclaimer():
    """Kiểm tra kết quả tổng hợp và sự kiện final luôn đi kèm lời miễn trừ trách nhiệm y tế."""
    profile = _load_sample_profile()
    orchestrator = Orchestrator()
    result = orchestrator.consult(profile)

    assert "disclaimer" in result
    assert "không phải chẩn đoán" in result["disclaimer"].lower()

    final_event = result["events"][-1]
    assert "disclaimer" in final_event
    assert final_event["disclaimer"] == result["disclaimer"]


def test_agents_do_not_call_each_other():
    """Kiểm tra các agent không import hoặc gọi trực tiếp nhau (kiểm tra tĩnh AST)."""
    agents_dir = ROOT / "src" / "diabetes_ai" / "agents"
    agent_files = {
        "risk_agent": agents_dir / "risk_agent.py",
        "safety_agent": agents_dir / "safety_agent.py",
        "knowledge_agent": agents_dir / "knowledge_agent.py",
    }

    for current_agent, filepath in agent_files.items():
        assert filepath.is_file(), f"Không tìm thấy file {filepath}"
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        # Kiểm tra không import các agent khác
        other_agents = [name for name in agent_files.keys() if name != current_agent]
        other_agents.append("orchestrator")

        for other in other_agents:
            for imp in imported_modules:
                assert other not in imp, (
                    f"Agent '{current_agent}' vi phạm nguyên tắc độc lập: đã import '{imp}'"
                )


def _safety_state(probs: dict[str, float], risk_index: float) -> dict:
    return {
        "profile": _load_sample_profile(),
        "risk_result": {"risk_level": "cao", "risk_index": risk_index, "probabilities": probs},
    }


def test_safety_agent_r10_uses_true_diabetes_probability():
    """R10 (p_diabetes >= 0.5) phải dựa trên P(tiểu đường), không phải chỉ số nguy cơ."""
    agent = SafetyAgent()
    # risk_index = 0.62 nhưng P(tiểu đường) chỉ 0.30 → R10 KHÔNG được kích hoạt
    low = agent.run(_safety_state({"không": 0.30, "tiền tiểu đường": 0.40, "tiểu đường": 0.30}, 0.62))
    assert "R10" not in [a.rule_id for a in low["alerts"]]
    # P(tiểu đường) = 0.55 → R10 phải kích hoạt
    high = agent.run(_safety_state({"không": 0.25, "tiền tiểu đường": 0.20, "tiểu đường": 0.55}, 0.65))
    assert "R10" in [a.rule_id for a in high["alerts"]]

