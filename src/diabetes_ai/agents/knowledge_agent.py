"""Truy xuất tri thức y khoa và tài liệu trích dẫn có nguồn gốc (Graph-RAG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diabetes_ai.knowledge.graph_rag import search


class KnowledgeAgent:
    """Tác tử truy xuất thông tin hướng dẫn y khoa và bằng chứng từ kho tri thức."""

    def __init__(
        self,
        chunks_path: Path | str | None = None,
        graph_path: Path | str | None = None,
    ) -> None:
        """Khởi tạo KnowledgeAgent.

        Args:
            chunks_path: Đường dẫn tới file chunks.json (tùy chọn).
            graph_path: Đường dẫn tới file graph.json (tùy chọn).
        """
        self.chunks_path = chunks_path
        self.graph_path = graph_path

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Truy xuất tài liệu trích dẫn theo triệu chứng và mức nguy cơ.

        Args:
            state: Từ điển trạng thái chứa 'profile' và 'risk_level' hoặc 'risk_result'.

        Returns:
            dict[str, Any]: Kết quả truy xuất gồm câu truy vấn và danh sách các đoạn trích dẫn.
        """
        profile = state.get("profile", state)
        risk_result = state.get("risk_result", {})
        risk_level = (
            state.get("risk_level")
            or risk_result.get("risk_level")
            or "tiểu đường"
        )

        profile_dict = profile if isinstance(profile, dict) else profile.model_dump()
        symptoms = profile_dict.get("symptoms", [])

        if "query" in state and state["query"]:
            query = str(state["query"])
        elif symptoms:
            query = f"{' '.join(symptoms)} nguy cơ {risk_level} đái tháo đường"
        else:
            query = f"nguy cơ {risk_level} đái tháo đường chỉ số xét nghiệm"

        citations = search(
            query=query,
            k=3,
            chunks_path=self.chunks_path,
            graph_path=self.graph_path,
        )

        return {
            "agent": "knowledge_agent",
            "query": query,
            "citations": citations,
        }
