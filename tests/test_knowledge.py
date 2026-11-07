"""Kiểm thử kho tri thức, thuật toán BM25 và truy xuất Graph-RAG."""

import json
from pathlib import Path

from diabetes_ai.knowledge.bm25 import BM25, tokens
from diabetes_ai.knowledge.build_kb import build
from diabetes_ai.knowledge.graph_rag import search


def test_bm25_ranks_exact_term_first():
    """Kiểm tra BM25 xếp hạng văn bản chứa từ khóa chính xác lên đầu tiên."""
    docs = [
        "Người trưởng thành cần duy trì chế độ dinh dưỡng lành mạnh và vận động thể lực thường xuyên.",
        "Tiêu chuẩn chẩn đoán xác định bệnh đái tháo đường khi chỉ số xét nghiệm HbA1c đạt từ 6.5% trở lên.",
        "Huyết áp tâm thu đo được ở mức dưới 130 mmHg và huyết áp tâm trương dưới 80 mmHg là tối ưu.",
    ]
    bm25 = BM25(docs)
    q_tokens = tokens("HbA1c")
    scores = bm25.scores(q_tokens)

    assert len(scores) == 3
    # Văn bản thứ hai (index 1) chứa trực tiếp thuật ngữ HbA1c, điểm số phải cao nhất
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    assert best_idx == 1


def test_graph_boost_changes_rank(tmp_path: Path):
    """Kiểm tra điểm thưởng từ đồ thị tri thức làm thay đổi thứ hạng tìm kiếm."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)

    # Chunk A lặp từ khóa nhiều lần nhưng không liên kết nút đồ thị mục tiêu
    # Chunk B có ít từ khóa hơn nhưng liên kết nút đồ thị 'dai_thao_duong'
    chunks = [
        {
            "id": "chunk_a",
            "title": "Theo dõi chỉ số thường quy",
            "text": "Kiểm tra sức khỏe định kỳ và thực hiện đo đường huyết, theo dõi đường huyết, kiểm soát đường huyết mỗi tháng.",
            "source": "Tài liệu phòng khám A",
            "nodes": [],
        },
        {
            "id": "chunk_b",
            "title": "Chẩn đoán bệnh lý chuyển hóa",
            "text": "Xét nghiệm đường huyết giúp phát hiện sớm và can thiệp kịp thời các hội chứng chuyển hóa.",
            "source": "Tài liệu chuyên khoa B",
            "nodes": ["dai_thao_duong"],
        },
    ]
    graph = {
        "nodes": [
            {"id": "dai_thao_duong", "label": "Đái tháo đường", "aliases": ["đái tháo đường", "tiểu đường"]},
        ],
        "edges": [],
    }

    chunks_file = processed_dir / "chunks.json"
    graph_file = processed_dir / "graph.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    # Khi không có boost (boost_match = 0.0), Chunk A xếp đầu do tần suất từ cao hơn
    res_no_boost = search(
        "tiểu đường đường huyết",
        k=2,
        boost_match=0.0,
        chunks_path=chunks_file,
        graph_path=graph_file,
    )
    assert res_no_boost[0]["id"] == "chunk_a"

    # Khi có boost nút đồ thị (boost_match = 5.0), Chunk B được cộng điểm và vươn lên vị trí thứ nhất
    res_boost = search(
        "tiểu đường đường huyết",
        k=2,
        boost_match=5.0,
        chunks_path=chunks_file,
        graph_path=graph_file,
    )
    assert res_boost[0]["id"] == "chunk_b"


def test_search_returns_source():
    """Kiểm tra tìm kiếm Graph-RAG trả về đoạn nguyên văn và nguồn gốc trích dẫn."""
    results = search("đường huyết đói", k=3)
    assert len(results) > 0

    for item in results:
        assert "text" in item and isinstance(item["text"], str) and len(item["text"].strip()) > 0
        assert "source" in item and isinstance(item["source"], str) and len(item["source"].strip()) > 0
        assert "score" in item and isinstance(item["score"], float)
        # Đảm bảo không sinh văn bản tùy tiện, đoạn trả về là văn bản gốc có tiêu đề
        assert "title" in item


def test_build_creates_files(tmp_path: Path):
    """Kiểm tra hàm build tự tạo file mẫu và ghi chunks.json cùng graph.json hợp lệ."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"

    summary = build(raw_dir=raw_dir, out_dir=out_dir)

    assert isinstance(summary, dict)
    assert (out_dir / "chunks.json").is_file()
    assert (out_dir / "graph.json").is_file()

    with open(out_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(out_dir / "graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    assert len(chunks) > 0
    assert "nodes" in graph and len(graph["nodes"]) > 0
    assert "edges" in graph and len(graph["edges"]) > 0

    # Kiểm tra mỗi chunk có các trường bắt buộc và độ dài từ 120-200 từ
    for c in chunks:
        assert "id" in c and "text" in c and "source" in c
        words = c["text"].split()
        assert 120 <= len(words) <= 200, f"Đoạn {c['id']} có độ dài {len(words)} từ, ngoài khoảng 120-200"
