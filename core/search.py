"""교재 검색. docs/sw/02_설계.md 결정 6·7·8.

낱말 검색(BM25)과 임베딩 검색에서 각각 상위 조각을 뽑아 합친다. 순위를 섞지 않는다.
임베딩을 못 쓰면(색인 없음, 질의 벡터 없음) 낱말 검색 결과만 돌려준다.
표준 라이브러리만 쓴다. 임베딩 API를 부르는 일은 core/llm.py가 한다.
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .kb import KBChunk

BM25_TOP = 10       # 분류에 넘기는 낱말 검색 상위 조각 수 (결정 8)
EMBEDDING_TOP = 10  # 분류에 넘기는 임베딩 검색 상위 조각 수


def tokens(text: str) -> List[str]:
    """한글은 두 글자씩, 영문·숫자는 낱말 단위로 자른다."""
    out: List[str] = []
    for word in re.findall(r"[가-힣]+|[a-z_]+|\d+", text.lower()):
        if "가" <= word[0] <= "힣":
            out += [word[i:i + 2] for i in range(len(word) - 1)] or [word]
        else:
            out.append(word)
    return out


class BM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.tf = [Counter(tokens(t)) for t in texts]
        self.length = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.length) / len(self.length)) if self.length else 1.0
        df = Counter(w for t in self.tf for w in t)
        n = len(texts)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> List[float]:
        q = Counter(tokens(query))
        out = []
        for tf, length in zip(self.tf, self.length):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out


def normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def text_key(model: str, text: str) -> str:
    """임베딩 색인 파일에서 조각을 찾는 열쇠. 모델이나 글이 바뀌면 달라진다."""
    return hashlib.sha256((model + "\n" + text).encode("utf-8")).hexdigest()


def _top(scores: Sequence[float], k: int, positive_only: bool = False) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return order[:k]


class Searcher:
    """교재 조각 색인. 서버를 띄울 때 한 번 만들고 요청마다 읽기만 한다."""

    def __init__(self, chunks: Sequence[KBChunk], vectors: Optional[Sequence[Sequence[float]]] = None):
        self.chunks = list(chunks)
        self.bm25 = BM25([c.text for c in self.chunks])
        # 조각과 같은 순서의, 길이 1로 맞춘 벡터. 없으면 낱말 검색만 한다
        self.vectors = [list(v) for v in vectors] if vectors is not None else None
        if self.vectors is not None and len(self.vectors) != len(self.chunks):
            self.vectors = None
        sections: Dict[str, List[KBChunk]] = {}
        for c in self.chunks:
            sections.setdefault(c.section_id, []).append(c)
        self._sections = sections

    @property
    def has_embeddings(self) -> bool:
        return self.vectors is not None

    def section_ids(self) -> List[str]:
        return list(self._sections)

    def section_text(self, section_id: str) -> str:
        """섹션 전체 글. 첫 조각의 섹션 제목 줄은 한 번만 넣는다."""
        parts = []
        for i, c in enumerate(self._sections.get(section_id, [])):
            text = c.text
            if i:
                text = text.split("\n", 1)[1] if "\n" in text else ""
            parts.append(text)
        return "\n\n".join(p for p in parts if p)

    def search(self, query: str, query_vector: Optional[Sequence[float]] = None,
               bm25_top: int = BM25_TOP, embedding_top: int = EMBEDDING_TOP) -> List[KBChunk]:
        """낱말 검색 상위와 임베딩 검색 상위를 합친다. 낱말 검색 결과가 앞에 온다."""
        picked = _top(self.bm25.scores(query), bm25_top, positive_only=True)
        if self.vectors is not None and query_vector is not None:
            q = normalize(query_vector)
            sims = [sum(x * y for x, y in zip(q, v)) for v in self.vectors]
            for i in _top(sims, embedding_top):
                if i not in picked:
                    picked.append(i)
        return [self.chunks[i] for i in picked]


# ---------- 임베딩 색인 파일 ----------

def load_vectors(path, model: str, chunks: Sequence[KBChunk]) -> Optional[List[List[float]]]:
    """색인 파일에서 조각 순서대로 벡터를 읽는다. 하나라도 없으면 None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    out = []
    for c in chunks:
        v = stored.get(text_key(model, c.text))
        if not isinstance(v, list):
            return None
        out.append(v)
    return out


def save_vectors(path, model: str, chunks: Sequence[KBChunk], vectors: Sequence[Sequence[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {text_key(model, c.text): [round(x, 6) for x in normalize(v)] for c, v in zip(chunks, vectors)}
    path.write_text(json.dumps(data), encoding="utf-8")
