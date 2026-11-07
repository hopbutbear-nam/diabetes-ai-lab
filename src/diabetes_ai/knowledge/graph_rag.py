"""Truy xuất tài liệu kết hợp đồ thị tri thức (Graph-RAG)."""

import json
from pathlib import Path
from typing import Any

from diabetes_ai.knowledge.bm25 import BM25, tokens
from diabetes_ai.knowledge.build_kb import build
from diabetes_ai.paths import ROOT


def search(
    query: str,
    k: int = 3,
    boost_match: float = 1.0,
    boost_neighbor: float = 0.3,
    chunks_path: Path | str | None = None,
    graph_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Tìm kiếm đoạn tri thức kết hợp điểm BM25 và tăng điểm nút đồ thị.

    Thuật toán:
    1. Xác định các nút trong đồ thị khớp với từ khóa truy vấn.
    2. Tìm tập các nút láng giềng bậc 1 liên kết với các nút khớp.
    3. Tính điểm BM25 giữa query và nội dung từng đoạn văn bản.
    4. Cộng điểm thưởng boost_match cho đoạn gắn với nút khớp trực tiếp,
       và boost_neighbor cho đoạn gắn với nút láng giềng.
    5. Trả về k đoạn có điểm cao nhất kèm văn bản nguyên văn và nguồn gốc. Không sinh văn bản tự do.

    Args:
        query: Chuỗi câu hỏi hoặc triệu chứng cần tra cứu.
        k: Số lượng đoạn tối đa trả về.
        boost_match: Điểm cộng cho đoạn gắn với nút khớp trực tiếp.
        boost_neighbor: Điểm cộng cho đoạn gắn với nút láng giềng.
        chunks_path: Đường dẫn tới file chunks.json (tùy chọn).
        graph_path: Đường dẫn tới file graph.json (tùy chọn).

    Returns:
        Danh sách dict chứa id, text (nguyên văn), source (nguồn) và điểm số.
    """
    c_path = Path(chunks_path) if chunks_path is not None else ROOT / "knowledge_base" / "processed" / "chunks.json"
    g_path = Path(graph_path) if graph_path is not None else ROOT / "knowledge_base" / "processed" / "graph.json"

    if not c_path.is_file() or not g_path.is_file():
        raw_dir = ROOT / "knowledge_base" / "raw"
        processed_dir = ROOT / "knowledge_base" / "processed"
        build(raw_dir=raw_dir, out_dir=processed_dir)

    with open(c_path, "r", encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)

    with open(g_path, "r", encoding="utf-8") as f:
        graph: dict[str, Any] = json.load(f)

    if not chunks:
        return []

    # 1. Khớp nút truy vấn với đồ thị
    q_norm = query.lower()
    q_tokens = tokens(query)
    q_token_set = set(q_tokens)
    matched_nodes: set[str] = set()

    for node in graph.get("nodes", []):
        node_id = node["id"]
        aliases = [node_id, node.get("label", "").lower()] + [a.lower() for a in node.get("aliases", [])]
        found = False
        for a in aliases:
            if not a:
                continue
            if a in q_norm:
                found = True
                break
            a_toks = tokens(a)
            if a_toks and set(a_toks).issubset(q_token_set):
                found = True
                break
        if found:
            matched_nodes.add(node_id)

    # 2. Tìm nút láng giềng bậc 1
    neighbor_nodes: set[str] = set()
    for edge in graph.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src in matched_nodes and tgt not in matched_nodes:
            neighbor_nodes.add(tgt)
        if tgt in matched_nodes and src not in matched_nodes:
            neighbor_nodes.add(src)

    # 3. Tính điểm BM25 cho các văn bản nguyên văn
    doc_texts = [c["text"] for c in chunks]
    bm25 = BM25(doc_texts)
    bm25_scores = bm25.scores(q_tokens)

    # 4. Kết hợp BM25 và tăng điểm đồ thị
    ranked_items: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        c_nodes = set(c.get("nodes", []))
        matched_hits = c_nodes & matched_nodes
        neighbor_hits = c_nodes & neighbor_nodes

        boost = 0.0
        if matched_hits:
            boost += boost_match
        elif neighbor_hits:
            boost += boost_neighbor

        final_score = float(bm25_scores[i]) + boost
        ranked_items.append(
            {
                "id": c.get("id", f"chunk_{i}"),
                "title": c.get("title", ""),
                "text": c["text"],
                "source": c.get("source", ""),
                "score": round(final_score, 4),
                "bm25_score": round(bm25_scores[i], 4),
                "matched_nodes": sorted(list(matched_hits)),
                "neighbor_nodes": sorted(list(neighbor_hits)),
            }
        )

    # Sắp xếp giảm dần theo điểm số tổng hợp, giải hòa bằng bm25_score
    ranked_items.sort(key=lambda item: (item["score"], item["bm25_score"]), reverse=True)
    return ranked_items[:k]
