"""Cắt đoạn tài liệu y khoa và dựng đồ thị tri thức (Graph-RAG)."""

import json
import re
from pathlib import Path
from typing import Any

from diabetes_ai.knowledge.bm25 import tokens

DEFAULT_GRAPH: dict[str, list[dict[str, Any]]] = {
    "nodes": [
        {
            "id": "dai_thao_duong",
            "label": "Đái tháo đường",
            "aliases": ["đái tháo đường", "tiểu đường", "diabetes", "tăng đường huyết"],
        },
        {
            "id": "tien_dai_thao_duong",
            "label": "Tiền đái tháo đường",
            "aliases": ["tiền đái tháo đường", "prediabetes", "rối loạn dung nạp glucose", "rối loạn đường huyết đói"],
        },
        {
            "id": "duong_huyet_doi",
            "label": "Đường huyết lúc đói",
            "aliases": ["đường huyết lúc đói", "đường huyết đói", "glucose đói", "fpg", "fasting glucose"],
        },
        {
            "id": "hba1c",
            "label": "HbA1c",
            "aliases": ["hba1c", "chỉ số hba1c", "hemoglobin a1c", "a1c"],
        },
        {
            "id": "dung_nap_glucose",
            "label": "Dung nạp glucose",
            "aliases": ["dung nạp glucose", "ogtt", "nghiệm pháp dung nạp glucose"],
        },
        {
            "id": "trieu_chung",
            "label": "Triệu chứng lâm sàng",
            "aliases": ["triệu chứng", "khát nước", "tiểu nhiều", "sụt cân", "ăn nhiều", "mệt mỏi", "symptoms"],
        },
        {
            "id": "huyet_ap",
            "label": "Huyết áp",
            "aliases": ["huyết áp", "tăng huyết áp", "huyết áp cao", "hypertension", "blood pressure"],
        },
        {
            "id": "tim_mach",
            "label": "Tim mạch",
            "aliases": ["tim mạch", "bệnh tim", "xơ vữa động mạch", "đột quỵ", "cardiovascular"],
        },
        {
            "id": "bmi",
            "label": "Chỉ số BMI",
            "aliases": ["bmi", "thừa cân", "béo phì", "chỉ số khối cơ thể", "cân nặng", "obesity", "overweight"],
        },
        {
            "id": "bien_chung",
            "label": "Biến chứng đái tháo đường",
            "aliases": ["biến chứng", "biến chứng thận", "biến chứng mắt", "võng mạc", "loét bàn chân", "complications"],
        },
        {
            "id": "loi_song",
            "label": "Lối sống và dinh dưỡng",
            "aliases": ["lối sống", "dinh dưỡng", "vận động", "tập thể dục", "chế độ ăn", "lifestyle"],
        },
    ],
    "edges": [
        {"source": "duong_huyet_doi", "target": "dai_thao_duong", "relation": "tiêu_chuẩn_chẩn_đoán"},
        {"source": "hba1c", "target": "dai_thao_duong", "relation": "tiêu_chuẩn_chẩn_đoán"},
        {"source": "dung_nap_glucose", "target": "dai_thao_duong", "relation": "tiêu_chuẩn_chẩn_đoán"},
        {"source": "duong_huyet_doi", "target": "tien_dai_thao_duong", "relation": "tiêu_chuẩn_chẩn_đoán"},
        {"source": "hba1c", "target": "tien_dai_thao_duong", "relation": "tiêu_chuẩn_chẩn_đoán"},
        {"source": "dung_nap_glucose", "target": "tien_dai_thao_duong", "relation": "tiêu_chuẩn_chẩn_đoán"},
        {"source": "tien_dai_thao_duong", "target": "dai_thao_duong", "relation": "tiến_triển_thành"},
        {"source": "trieu_chung", "target": "dai_thao_duong", "relation": "chỉ_điểm_lâm_sàng"},
        {"source": "bmi", "target": "tien_dai_thao_duong", "relation": "yếu_tố_nguy_cơ"},
        {"source": "bmi", "target": "dai_thao_duong", "relation": "yếu_tố_nguy_cơ"},
        {"source": "huyet_ap", "target": "tim_mach", "relation": "yếu_tố_nguy_cơ"},
        {"source": "dai_thao_duong", "target": "tim_mach", "relation": "tăng_nguy_cơ"},
        {"source": "dai_thao_duong", "target": "bien_chung", "relation": "dẫn_đến"},
        {"source": "loi_song", "target": "tien_dai_thao_duong", "relation": "phòng_ngừa"},
        {"source": "loi_song", "target": "dai_thao_duong", "relation": "kiểm_soát"},
    ],
}

SAMPLE_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "chunk_0",
        "title": "Tiêu chuẩn chẩn đoán bệnh đái tháo đường theo ADA 2024",
        "source": "ADA Standards of Care in Diabetes 2024 - Section 2: Diagnosis",
        "nodes": ["dai_thao_duong", "duong_huyet_doi", "hba1c", "trieu_chung"],
        "text": (
            "Theo Tiêu chuẩn Chăm sóc Y tế trong Đái tháo đường của Hiệp hội Đái tháo đường Hoa Kỳ (ADA 2024), "
            "bệnh đái tháo đường được xác định bằng một trong các tiêu chí cận lâm sàng sau đây: Thứ nhất, nồng độ glucose huyết "
            "tương lúc đói (FPG) đạt từ 126 mg/dL (tương đương 7,0 mmol/L) trở lên sau khi nhịn ăn ít nhất 8 giờ. Thứ hai, chỉ số "
            "hemoglobin glycated (HbA1c) đạt từ 6,5% (48 mmol/mol) trở lên, được thực hiện bằng phương pháp xét nghiệm chuẩn hóa theo "
            "chương trình NGSP. Thứ ba, mức glucose huyết tương sau 2 giờ làm nghiệm pháp dung nạp glucose đường uống (OGTT với liều nạp "
            "75g glucose) đạt từ 200 mg/dL (11,1 mmol/L) trở lên. Thứ tư, ở người bệnh có các triệu chứng kinh điển của tăng đường "
            "huyết hoặc cơn tăng đường huyết cấp tính, nồng độ glucose huyết tương ngẫu nhiên bất kỳ đạt từ 200 mg/dL trở lên đủ "
            "tiêu chuẩn kết luận mà không cần lặp lại xét nghiệm lần thứ hai."
        ),
    },
    {
        "id": "chunk_1",
        "title": "Tiêu chuẩn chẩn đoán tiền đái tháo đường theo ADA 2024",
        "source": "ADA Standards of Care in Diabetes 2024 - Section 2: Diagnosis",
        "nodes": ["tien_dai_thao_duong", "duong_huyet_doi", "hba1c", "dung_nap_glucose"],
        "text": (
            "Tiền đái tháo đường là giai đoạn trung gian giữa chuyển hóa glucose bình thường và đái tháo đường, mang nguy cơ cao "
            "tiến triển thành bệnh mạn tính cùng biến chứng tim mạch sớm. Theo hướng dẫn cập nhật của ADA 2024, tình trạng tiền đái tháo "
            "đường được chẩn đoán khi thỏa mãn một trong ba tiêu chuẩn sau: Một là rối loạn đường huyết lúc đói (IFG) với chỉ số "
            "glucose huyết tương khi nhịn ăn dao động trong khoảng từ 100 mg/dL đến 125 mg/dL (tương đương 5,6 đến 6,9 mmol/L). "
            "Hai là rối loạn dung nạp glucose (IGT) với mức glucose huyết tương đo được sau 2 giờ làm nghiệm pháp dung nạp 75g glucose "
            "đường uống nằm trong khoảng từ 140 mg/dL đến 199 mg/dL (tương đương 7,8 đến 11,0 mmol/L). Ba là chỉ số HbA1c nằm trong "
            "khoảng từ 5,7% đến 6,4% (39 đến 47 mmol/mol). Việc phát hiện sớm ở giai đoạn này tạo cơ hội vàng để đảo ngược bệnh."
        ),
    },
    {
        "id": "chunk_2",
        "title": "Triệu chứng lâm sàng cảnh báo và dấu hiệu nguy hiểm của đái tháo đường",
        "source": "ADA Standards of Care in Diabetes 2024 & NIDDK Clinical Symptoms",
        "nodes": ["trieu_chung", "dai_thao_duong", "duong_huyet_doi"],
        "text": (
            "Các triệu chứng lâm sàng kinh điển của đái tháo đường thường được mô tả qua hội chứng bốn nhiều bao gồm: ăn nhiều "
            "do tế bào thiếu hụt năng lượng, uống nhiều và khát nước liên tục do mất nước tế bào, tiểu nhiều đặc biệt vào ban đêm do "
            "lợi niệu thẩm thấu khi nồng độ glucose trong máu vượt ngưỡng tái hấp thu của thận, cùng với sụt cân nhanh chóng không rõ "
            "nguyên nhân dù khẩu phần ăn không hề giảm sút. Ngoài ra, người bệnh có thể xuất hiện cảm giác mệt mỏi suy nhược kéo dài, "
            "thị lực giảm sút mờ thoáng qua, cảm giác tê bì dị cảm ở bàn chân hoặc vết thương chậm liền sẹo. Theo ADA 2024, khi xuất hiện "
            "triệu chứng nguy hiểm như nôn mửa liên tục, thở nhanh sâu có mùi ceton táo thối, rối loạn tri giác li bì hoặc glucose huyết "
            "tương tăng vọt trên 250 mg/dL, cần lập tức đưa bệnh nhân đến cơ sở y tế cấp cứu khẩn cấp."
        ),
    },
    {
        "id": "chunk_3",
        "title": "Kiểm soát huyết áp và giảm thiểu nguy cơ tim mạch ở người đái tháo đường",
        "source": "ADA Standards of Care in Diabetes 2024 - Section 10: Cardiovascular Disease",
        "nodes": ["huyet_ap", "tim_mach", "dai_thao_duong", "bien_chung"],
        "text": (
            "Bệnh tim mạch xơ vữa là nguyên nhân hàng đầu gây tàn tật và tử vong ở bệnh nhân đái tháo đường type 2. Theo hướng dẫn "
            "thực hành lâm sàng ADA 2024, mục tiêu kiểm soát huyết áp được khuyến cáo cho hầu hết bệnh nhân đái tháo đường kèm tăng huyết "
            "áp là đạt mức huyết áp tâm thu dưới 130 mmHg và huyết áp tâm trương dưới 80 mmHg khi có thể đạt được một cách an toàn. Chiến "
            "lược kiểm soát huyết áp cần khởi đầu bằng chế độ ăn giảm muối DASH, tăng cường hoạt động thể chất và điều trị dược lý với "
            "các nhóm thuốc có lợi ích bảo vệ thận và tim mạch đã được chứng minh như thuốc ức chế men chuyển hoặc thuốc chẹn thụ thể "
            "angiotensin. Bên cạnh đó, ADA khuyến nghị chỉ định liệu pháp statin dự phòng nguyên phát cho người bệnh đái tháo đường "
            "trong độ tuổi 40 đến 75 để tối ưu hóa nồng độ mỡ máu và giảm thiểu biến cố mạch vành."
        ),
    },
    {
        "id": "chunk_4",
        "title": "Quản lý cân nặng và chỉ số BMI trong phòng ngừa đái tháo đường",
        "source": "ADA Standards of Care in Diabetes 2024 - Section 8: Obesity Management",
        "nodes": ["bmi", "tien_dai_thao_duong", "dai_thao_duong", "loi_song"],
        "text": (
            "Tình trạng thừa cân và béo phì là yếu tố nguy cơ có thể thay đổi quan trọng nhất liên quan trực tiếp đến kháng insulin "
            "và đái tháo đường type 2. Hướng dẫn của ADA 2024 khuyến cáo thực hiện tầm soát đái tháo đường ở tất cả người trưởng thành không "
            "có triệu chứng nhưng có chỉ số khối cơ thể BMI từ 25 kg/m² trở lên (hoặc ngưỡng thấp hơn là từ 23 kg/m² trở lên đối với người "
            "gốc châu Á) kèm theo một hoặc nhiều yếu tố nguy cơ bổ sung như tăng huyết áp hay ít vận động. Mục tiêu can thiệp cân nặng "
            "trọng tâm là đạt được mức giảm cân tối thiểu từ 5% đến 7% so với trọng lượng ban đầu và duy trì lâu dài. Các thử nghiệm lâm "
            "sàng quy mô lớn đã chứng minh rằng mức giảm cân khiêm tốn này kết hợp hoạt động thể chất đều đặn giúp giảm tới 58% nguy cơ "
            "tiến triển thành đái tháo đường thực sự."
        ),
    },
    {
        "id": "chunk_5",
        "title": "Tầm soát và phát hiện sớm các biến chứng mạn tính của đái tháo đường",
        "source": "ADA Standards of Care in Diabetes 2024 - Section 11 & 12: Microvascular Complications",
        "nodes": ["bien_chung", "dai_thao_duong", "tim_mach"],
        "text": (
            "Tình trạng tăng đường huyết mạn tính không được kiểm soát tốt sẽ dẫn tới tổn thương vi mạch và mạch máu lớn trên khắp "
            "cơ thể, gây ra nhiều biến chứng nghiêm trọng đe dọa sức khỏe. Theo khuyến nghị của ADA 2024, người bệnh đái tháo đường cần "
            "được khám tầm soát biến chứng định kỳ tối thiểu mỗi năm một lần. Đánh giá tổn thương thận đái tháo đường được thực hiện bằng "
            "cách xét nghiệm tỷ lệ albumin trên creatinin niệu (UACR) và ước tính mức lọc cầu thận (eGFR). Tầm soát biến chứng võng mạc "
            "đái tháo đường cần được tiến hành bởi bác sĩ chuyên khoa mắt thông qua kỹ thuật soi đáy mắt có giãn đồng tử hoặc chụp ảnh màu "
            "võng mạc. Ngoài ra, việc kiểm tra bàn chân toàn diện hàng năm giúp đánh giá cảm giác bảo vệ của hệ thần kinh ngoại biên và lưu "
            "thông mạch máu, từ đó phòng ngừa loét hoại tử và nguy cơ cắt cụt chi."
        ),
    },
    {
        "id": "chunk_6",
        "title": "Khuyến nghị dinh dưỡng và vận động thể lực cho người bệnh đái tháo đường",
        "source": "ADA Standards of Care in Diabetes 2024 - Section 5: Lifestyle Management",
        "nodes": ["loi_song", "dai_thao_duong", "tien_dai_thao_duong"],
        "text": (
            "Liệu pháp dinh dưỡng y học và hoạt động thể lực là trụ cột cơ bản không thể thiếu trong quản lý bệnh đái tháo đường "
            "và tiền đái tháo đường theo ADA 2024. Về mặt vận động, khuyến cáo người trưởng thành cần tích lũy ít nhất 150 phút hoạt "
            "động thể lực cường độ trung bình mỗi tuần (như đi bộ nhanh, bơi lội, đạp xe hoặc khiêu vũ), trải đều ít nhất ba ngày trong "
            "tuần và không nghỉ quá hai ngày liên tiếp giữa các buổi tập. Người bệnh nên kết hợp tập rèn luyện sức bền hoặc kháng lực từ "
            "hai đến ba buổi mỗi tuần. Về dinh dưỡng, khuyến nghị ưu tiên các loại thực phẩm nguyên phần giàu chất xơ, rau củ quả, ngũ cốc "
            "nguyên hạt, các loại hạt đậu và nguồn chất béo không bão hòa đơn lành mạnh; đồng thời cắt giảm tối đa các loại đồ uống có "
            "đường, carbohydrate tinh chế cùng chất béo bão hòa có hại."
        ),
    },
]


def _detect_nodes(text: str, graph: dict[str, Any]) -> list[str]:
    """Phát hiện các nút khái niệm trong văn bản dựa vào id, nhãn và bí danh."""
    text_norm = text.lower()
    t_set = set(tokens(text))
    matched: list[str] = []
    for node in graph.get("nodes", []):
        node_id = node["id"]
        aliases = [node_id, node.get("label", "").lower()] + [a.lower() for a in node.get("aliases", [])]
        found = False
        for a in aliases:
            if not a:
                continue
            if a in text_norm or (tokens(a) and set(tokens(a)).issubset(t_set)):
                found = True
                break
        if found:
            matched.append(node_id)
    return sorted(list(set(matched)))


def _chunk_text(text: str, min_words: int = 120, max_words: int = 200) -> list[str]:
    """Cắt văn bản dài thành các đoạn từ 120 đến 200 từ theo câu."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if not s_words:
            continue
        if current_word_count + s_words > max_words and current_word_count >= min_words:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_word_count = s_words
        else:
            current_sentences.append(sentence)
            current_word_count += s_words

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def build(raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Xây dựng kho tri thức từ file raw, cắt đoạn và ghi ra processed.

    Args:
        raw_dir: Thư mục chứa các tài liệu thô (*.txt).
        out_dir: Thư mục lưu kết quả xử lý (chunks.json và graph.json).

    Returns:
        Dict tổng kết số lượng chunks, nodes, edges và đường dẫn file đã tạo.
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_path.glob("*.txt"))
    chunks: list[dict[str, Any]] = []

    if not txt_files:
        # Nếu thư mục rỗng, ghi file mẫu và nạp SAMPLE_CHUNKS chuẩn hóa
        sample_file = raw_path / "ada_guidelines_2024.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            for sc in SAMPLE_CHUNKS:
                f.write(f"### {sc['title']}\n")
                f.write(f"Nguồn: {sc['source']}\n\n")
                f.write(f"{sc['text']}\n\n---\n\n")

        for sc in SAMPLE_CHUNKS:
            chunk_copy = dict(sc)
            # Đảm bảo có liên kết nút đồ thị
            if not chunk_copy.get("nodes"):
                chunk_copy["nodes"] = _detect_nodes(chunk_copy["text"], DEFAULT_GRAPH)
            chunks.append(chunk_copy)
    else:
        # Đọc và phân đoạn các file người dùng/agent đặt vào
        chunk_idx = 0
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            source_name = f"Tài liệu {txt_file.stem}"
            sections = [sec.strip() for sec in content.split("---") if sec.strip()]
            if not sections:
                sections = [content.strip()]

            for sec in sections:
                # Kiểm tra tiêu đề hoặc nguồn tùy chọn trong phần
                title = f"Tri thức {txt_file.stem} #{chunk_idx + 1}"
                src = source_name
                lines = sec.splitlines()
                body_lines = []
                for line in lines:
                    if line.startswith("### "):
                        title = line.replace("### ", "").strip()
                    elif line.startswith("Nguồn: "):
                        src = line.replace("Nguồn: ", "").strip()
                    else:
                        body_lines.append(line)

                body_text = " ".join(" ".join(body_lines).split())
                if not body_text:
                    continue

                text_chunks = _chunk_text(body_text, min_words=120, max_words=200)
                for tc in text_chunks:
                    c_dict = {
                        "id": f"chunk_{chunk_idx}",
                        "title": title,
                        "source": src,
                        "text": tc,
                        "nodes": _detect_nodes(tc, DEFAULT_GRAPH),
                    }
                    chunks.append(c_dict)
                    chunk_idx += 1

    chunks_file = out_path / "chunks.json"
    graph_file = out_path / "graph.json"

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_GRAPH, f, ensure_ascii=False, indent=2)

    return {
        "chunks": len(chunks),
        "nodes": len(DEFAULT_GRAPH["nodes"]),
        "edges": len(DEFAULT_GRAPH["edges"]),
        "chunks_path": str(chunks_file),
        "graph_path": str(graph_file),
    }
