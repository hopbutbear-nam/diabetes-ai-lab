"""Thuật toán truy xuất BM25 thuần Python hỗ trợ tiếng Việt."""

import math
import re
import unicodedata
from collections import Counter


def tokens(text: str) -> list[str]:
    """Tách từ: chữ thường, bỏ dấu câu, giữ ký tự tiếng Việt và số.

    Args:
        text: Chuỗi văn bản đầu vào.

    Returns:
        Danh sách các token từ hợp lệ.
    """
    if not text:
        return []
    norm_text = unicodedata.normalize("NFC", text.lower())
    cleaned = norm_text.replace("_", " ")
    return re.findall(r"[^\W_]+", cleaned, flags=re.UNICODE)


class BM25:
    """Mô hình xếp hạng tài liệu Okapi BM25 thuần Python."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        """Khởi tạo BM25 với tập tài liệu và siêu tham số k1, b.

        Args:
            docs: Danh sách các văn bản gốc.
            k1: Tham số điều chỉnh độ bão hòa tần suất từ (mặc định 1.5).
            b: Tham số chuẩn hóa độ dài tài liệu (mặc định 0.75).
        """
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_size = len(docs)
        self.doc_tokens: list[list[str]] = [tokens(d) for d in docs]
        self.doc_lens: list[int] = [len(dt) for dt in self.doc_tokens]
        self.avgdl: float = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 1.0

        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs: list[Counter[str]] = []
        for dt in self.doc_tokens:
            tf = Counter(dt)
            self.term_freqs.append(tf)
            for word in tf.keys():
                self.doc_freqs[word] += 1

        self.idf: dict[str, float] = {}
        for word, df in self.doc_freqs.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str] | str) -> list[float]:
        """Tính điểm tương đồng BM25 giữa query và từng tài liệu trong kho.

        Args:
            query_tokens: Danh sách các token truy vấn hoặc chuỗi truy vấn.

        Returns:
            Danh sách điểm BM25 cho từng tài liệu tương ứng.
        """
        if self.corpus_size == 0:
            return []

        if isinstance(query_tokens, str):
            q_list = tokens(query_tokens)
        else:
            q_list = [t.lower() for t in query_tokens]

        if not q_list:
            return [0.0] * self.corpus_size

        scores: list[float] = [0.0] * self.corpus_size
        for q in q_list:
            if q not in self.idf:
                continue
            idf_val = self.idf[q]
            for i, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(q, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                    scores[i] += idf_val * (tf * (self.k1 + 1.0)) / denom

        return scores
