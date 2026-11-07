"""Điều phối hệ đa agent (Risk -> Safety -> Knowledge) theo luồng sự kiện chuẩn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from diabetes_ai.agents.knowledge_agent import KnowledgeAgent
from diabetes_ai.agents.risk_agent import RiskAgent
from diabetes_ai.agents.safety_agent import SafetyAgent
from diabetes_ai.paths import CONFIGS_DIR
from diabetes_ai.safety.critic import LEVEL_RANK
from diabetes_ai.safety.disclaimers import DISCLAIMER_VI, advice_for
from diabetes_ai.safety.risk_level import combine


class Orchestrator:
    """Bộ điều phối trung tâm quản lý luồng trao đổi và sự kiện giữa các tác tử."""

    def __init__(
        self,
        risk_agent: RiskAgent | None = None,
        safety_agent: SafetyAgent | None = None,
        knowledge_agent: KnowledgeAgent | None = None,
        config_path: Path | str | None = None,
        risk_levels: dict[str, float] | None = None,
    ) -> None:
        """Khởi tạo Orchestrator với các tác tử thành phần.

        Args:
            risk_agent: Tác tử suy luận nguy cơ (nếu None sẽ khởi tạo mặc định).
            safety_agent: Tác tử kiểm tra an toàn và phản biện (nếu None sẽ khởi tạo mặc định).
            knowledge_agent: Tác tử truy xuất tri thức (nếu None sẽ khởi tạo mặc định).
            config_path: Đường dẫn cấu hình phục vụ (mặc định configs/serve.yaml).
            risk_levels: Từ điển ngưỡng phân tầng nguy cơ {'low': float, 'medium': float}.
        """
        self.risk_agent = risk_agent or RiskAgent()
        self.safety_agent = safety_agent or SafetyAgent()
        self.knowledge_agent = knowledge_agent or KnowledgeAgent()

        self.low_th = 0.10
        self.med_th = 0.25

        if risk_levels is not None:
            self.low_th = float(risk_levels.get("low", self.low_th))
            self.med_th = float(risk_levels.get("medium", self.med_th))
        else:
            c_path = Path(config_path) if config_path else CONFIGS_DIR / "serve.yaml"
            if c_path.is_file():
                try:
                    with open(c_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                    rl = cfg.get("risk_levels", {})
                    self.low_th = float(rl.get("low", self.low_th))
                    self.med_th = float(rl.get("medium", self.med_th))
                except Exception:
                    pass

    def consult(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Tiến hành chu trình tư vấn đa tác tử theo thứ tự tuần tự risk -> safety -> knowledge.

        Sinh chuỗi sự kiện tuân thủ giao thức: speak | handoff | tool | cite | alert | final.

        Args:
            profile: Hồ sơ lâm sàng và lối sống của người dùng.

        Returns:
            dict[str, Any]: Toàn bộ dữ liệu phiên tư vấn gồm chuỗi events, phân tầng nguy cơ
                            tổng hợp, danh sách cảnh báo, trích dẫn tri thức và lời từ chối trách nhiệm.
        """
        events: list[dict[str, Any]] = []

        # =========================================================================
        # 1. RISK AGENT
        # =========================================================================
        events.append({
            "event": "handoff",
            "type": "handoff",
            "from": "orchestrator",
            "to": "risk_agent",
            "message": "Chuyển giao hồ sơ bệnh nhân cho RiskAgent để ước tính nguy cơ ban đầu.",
        })

        events.append({
            "event": "tool",
            "type": "tool",
            "agent": "risk_agent",
            "action": "predict",
            "input": profile,
        })

        risk_output = self.risk_agent.run({"profile": profile})

        events.append({
            "event": "speak",
            "type": "speak",
            "agent": "risk_agent",
            "content": (
                f"Chỉ số nguy cơ sơ bộ là {risk_output.get('risk_index', 0.0):.2f} "
                f"(mức: {risk_output.get('risk_level', 'thấp')})."
            ),
            "data": {
                "risk_index": risk_output.get("risk_index"),
                "risk_level": risk_output.get("risk_level"),
                "probabilities": risk_output.get("probabilities"),
            },
        })

        # =========================================================================
        # 2. SAFETY AGENT
        # =========================================================================
        events.append({
            "event": "handoff",
            "type": "handoff",
            "from": "orchestrator",
            "to": "safety_agent",
            "message": "Chuyển giao kết quả sang SafetyAgent để kiểm tra luật an toàn và phản biện chéo.",
        })

        events.append({
            "event": "tool",
            "type": "tool",
            "agent": "safety_agent",
            "action": "evaluate_and_cross_check",
        })

        safety_output = self.safety_agent.run({
            "profile": profile,
            "risk_result": risk_output,
        })

        raw_alerts = safety_output.get("alerts", [])
        alerts_data: list[dict[str, Any]] = []
        for alert in raw_alerts:
            a_dict = {
                "rule_id": getattr(alert, "rule_id", ""),
                "level": getattr(alert, "level", "warn"),
                "text": getattr(alert, "text", ""),
                "evidence": getattr(alert, "evidence", {}),
            }
            alerts_data.append(a_dict)
            events.append({
                "event": "alert",
                "type": "alert",
                "agent": "safety_agent",
                "level": a_dict["level"],
                "rule_id": a_dict["rule_id"],
                "message": a_dict["text"],
                "text": a_dict["text"],
            })

        for contradiction in safety_output.get("contradictions", []):
            events.append({
                "event": "alert",
                "type": "alert",
                "agent": "safety_agent",
                "level": "warn",
                "rule_id": contradiction,
                "message": f"Phản biện lâm sàng phát hiện mâu thuẫn ({contradiction}).",
                "text": f"Phản biện lâm sàng phát hiện mâu thuẫn ({contradiction}).",
            })

        events.append({
            "event": "speak",
            "type": "speak",
            "agent": "safety_agent",
            "content": (
                f"Hoàn thành rà soát an toàn lâm sàng: phát hiện {len(raw_alerts)} cảnh báo luật "
                f"và {len(safety_output.get('contradictions', []))} điểm mâu thuẫn phản biện."
            ),
        })

        # =========================================================================
        # TỔNG HỢP MỨC NGUY CƠ (Luật danger luôn nâng mức)
        # =========================================================================
        r_idx = float(risk_output.get("risk_index", 0.0))
        combined_lvl = combine(
            r=r_idx,
            alerts=raw_alerts,
            low=self.low_th,
            medium=self.med_th,
        )

        suggested_lvl = safety_output.get("suggested_level")
        if suggested_lvl and LEVEL_RANK.get(suggested_lvl, 0) > LEVEL_RANK.get(combined_lvl, 0):
            final_level = suggested_lvl
        else:
            final_level = combined_lvl

        # Đảm bảo luật: cờ danger luôn nâng mức lên cao
        if safety_output.get("has_danger"):
            final_level = "cao"

        # =========================================================================
        # 3. KNOWLEDGE AGENT
        # =========================================================================
        events.append({
            "event": "handoff",
            "type": "handoff",
            "from": "orchestrator",
            "to": "knowledge_agent",
            "message": "Chuyển giao sang KnowledgeAgent để tìm kiếm trích dẫn và tài liệu hướng dẫn lâm sàng.",
        })

        events.append({
            "event": "tool",
            "type": "tool",
            "agent": "knowledge_agent",
            "action": "graph_rag_search",
            "query_context": final_level,
        })

        knowledge_output = self.knowledge_agent.run({
            "profile": profile,
            "risk_result": risk_output,
            "risk_level": final_level,
        })

        citations = knowledge_output.get("citations", [])
        for c in citations:
            events.append({
                "event": "cite",
                "type": "cite",
                "agent": "knowledge_agent",
                "title": c.get("title", ""),
                "source": c.get("source", ""),
                "text": c.get("text", ""),
                "score": float(c.get("score", 0.0)),
            })

        events.append({
            "event": "speak",
            "type": "speak",
            "agent": "knowledge_agent",
            "content": f"Đã trích dẫn {len(citations)} tài liệu y khoa từ hướng dẫn chính thức.",
        })

        # =========================================================================
        # 4. FINAL EVENT
        # =========================================================================
        advice = advice_for(final_level, alerts=raw_alerts)
        final_event = {
            "event": "final",
            "type": "final",
            "agent": "orchestrator",
            "risk_level": final_level,
            "final_level": final_level,
            "risk_index": r_idx,
            "probabilities": risk_output.get("probabilities", {}),
            "alerts": alerts_data,
            "citations": citations,
            "advice": advice,
            "disclaimer": DISCLAIMER_VI,
        }
        events.append(final_event)

        return {
            "events": events,
            "risk_level": final_level,
            "final_level": final_level,
            "risk_index": r_idx,
            "probabilities": risk_output.get("probabilities", {}),
            "alerts": alerts_data,
            "citations": citations,
            "advice": advice,
            "disclaimer": DISCLAIMER_VI,
        }
