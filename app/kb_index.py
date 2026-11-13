"""Hybrid KB index (Kiwi + BM25s + optional OpenAI dense embeddings) and exemplar store."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import bm25s
import numpy as np
from kiwipiepy import Kiwi
from openai import AsyncOpenAI

from app.config import Settings
from app.normalize import collapse_korean_obfuscation, normalize_text

logger = logging.getLogger(__name__)

_TABLE_PLACEHOLDER_RE = re.compile(r"\(표:\s*([A-Za-z0-9_-]+-table\d+\.csv)\)")
_HEADING_SPLIT_RE = re.compile(r"(?m)^(#{2,3}\s+.+)$")
_EN_CODE_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")

_COURSE_TERMS: tuple[str, ...] = (
    "다람쥐은행",
    "다람 문구점",
    "다람문구점",
    "꿈틀이",
    "꿈틀",
    "kkumtle",
    "Helpy",
    "HelpyCode",
    "첫걸음 통장",
    "첫걸음통장",
    "새싹적금",
    "꿈모으기 정기예금",
    "틴즈 체크카드",
    "도토리 포인트",
    "도토리포인트",
    "하람",
    "솔숲 쉼터",
    "거름망",
    "바이브 코딩",
    "fruit_basket",
    "fruit_order",
    "용돈 기입장",
    "바른말 필터",
)

_DOMAIN_MAP: dict[str, str] = {
    "FIN": "finance",
    "PR": "prompt",
    "PY": "python",
}


@dataclass(frozen=True)
class ParentSection:
    section_id: str
    domain: str
    title: str
    raw_text: str
    table_csvs: dict[str, list[dict[str, str]]] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    section_id: str
    title: str
    text: str
    is_table: bool = False


@dataclass(frozen=True)
class RetrievedSection:
    section_id: str
    title: str
    full_text: str
    rrf_score: float
    bm25_score: float
    dense_score: float


def _build_kiwi() -> Kiwi:
    kiwi = Kiwi()
    for term in _COURSE_TERMS:
        try:
            kiwi.add_user_word(term, "NNP", 0)
        except Exception:
            pass
    return kiwi


def _format_csv_markdown(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _read_csv_table(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        all_rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not all_rows:
        return [], []
    headers = [h.strip() for h in all_rows[0]]
    rows = [[c.strip() for c in r] for r in all_rows[1:]]
    return headers, rows


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)


class KBIndex:
    """Hybrid sparse (Kiwi + BM25s) and dense (OpenAI embeddings) KB retriever."""

    def __init__(self, settings: Settings, embed_client: Optional[AsyncOpenAI] = None):
        self.settings = settings
        self.embed_client = embed_client
        self._kiwi = _build_kiwi()
        self._embed_lock = asyncio.Lock()

        self.sections: dict[str, ParentSection] = {}
        self.section_order: list[str] = []
        self.chunks: list[ChildChunk] = []
        self._chunk_id_to_idx: dict[str, int] = {}

        self._contextual_headers: dict[str, str] = {}
        self._chunk_embeddings: Optional[np.ndarray] = None

        self.tutor_examples: list[dict[str, Any]] = []
        self.guard_examples: list[dict[str, Any]] = []
        self._tutor_embeddings: Optional[np.ndarray] = None

        self._load_kb()
        self._load_public_examples()
        self._load_precomputed_assets()
        self._build_bm25_indices()

    def _tokenize(self, text: str) -> list[str]:
        cleaned = collapse_korean_obfuscation(text)
        if not cleaned:
            return []
        tokens: list[str] = []
        for tok in self._kiwi.tokenize(cleaned):
            if tok.tag.startswith("S"):
                continue
            form = tok.form.strip().lower()
            if form:
                tokens.append(form)
        for m in _EN_CODE_TOKEN_RE.findall(cleaned):
            low = m.lower()
            if low not in tokens:
                tokens.append(low)
        return tokens

    def _load_kb(self) -> None:
        kb_dir = self.settings.kb_dir
        if not kb_dir.exists():
            logger.warning("KB directory does not exist: %s", kb_dir)
            return

        md_files = sorted(
            p for p in kb_dir.rglob("*.md") if p.stem.startswith(("FIN-", "PR-", "PY-"))
        )

        for md_path in md_files:
            section_id = md_path.stem
            prefix = section_id.split("-", 1)[0]
            domain = _DOMAIN_MAP.get(prefix, md_path.parent.name)
            raw_md = md_path.read_text(encoding="utf-8")

            lines = raw_md.splitlines()
            title = section_id
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# "):
                    heading_text = stripped[2:].strip()
                    if heading_text.startswith(section_id):
                        heading_text = heading_text[len(section_id) :].strip()
                    title = heading_text or section_id
                    break

            # Load any matching CSV tables in the same directory
            table_csvs: dict[str, list[dict[str, str]]] = {}
            table_markdowns: dict[str, str] = {}
            for csv_path in sorted(md_path.parent.glob(f"{section_id}-table*.csv")):
                headers, rows = _read_csv_table(csv_path)
                if not headers:
                    continue
                dict_rows = [
                    {
                        headers[i]: (row[i] if i < len(row) else "")
                        for i in range(len(headers))
                    }
                    for row in rows
                ]
                table_csvs[csv_path.name] = dict_rows
                table_markdowns[csv_path.name] = _format_csv_markdown(headers, rows)

                # Verbalize each row into a natural language child chunk
                for row_idx, row in enumerate(rows, start=1):
                    pairs = ", ".join(
                        f"{headers[i]}: {row[i] if i < len(row) else ''}"
                        for i in range(len(headers))
                    )
                    chunk_id = f"{section_id}::table::{csv_path.name}::r{row_idx}"
                    chunk_text = f"[표 {section_id} {title}] {pairs}"
                    self._add_chunk(
                        ChildChunk(
                            chunk_id=chunk_id,
                            section_id=section_id,
                            title=title,
                            text=chunk_text,
                            is_table=True,
                        )
                    )

            # Replace (표: {sid}-table1.csv) placeholders or append unreferenced tables
            used_tables: set[str] = set()

            def _replace_table(match: re.Match[str]) -> str:
                fname = match.group(1)
                if fname in table_markdowns:
                    used_tables.add(fname)
                    return f"(표: {fname})\n{table_markdowns[fname]}"
                return match.group(0)

            enriched_text = _TABLE_PLACEHOLDER_RE.sub(_replace_table, raw_md)
            for fname, md_table in table_markdowns.items():
                if fname not in used_tables:
                    enriched_text += f"\n\n[표: {fname}]\n{md_table}"

            self.sections[section_id] = ParentSection(
                section_id=section_id,
                domain=domain,
                title=title,
                raw_text=enriched_text,
                table_csvs=table_csvs,
            )
            self.section_order.append(section_id)

            # Split markdown into child chunks on ## / ### headings
            self._split_md_chunks(section_id, title, enriched_text)

    def _add_chunk(self, chunk: ChildChunk) -> None:
        self._chunk_id_to_idx[chunk.chunk_id] = len(self.chunks)
        self.chunks.append(chunk)

    def _split_md_chunks(self, section_id: str, title: str, text: str) -> None:
        parts = _HEADING_SPLIT_RE.split(text)
        # parts[0] is content before first ##/### heading (usually # Title + intro)
        intro = parts[0].strip()
        if intro:
            # Strip the top-level # heading line if it's alone, keep any body
            intro_lines = [
                ln for ln in intro.splitlines() if not ln.strip().startswith("# ")
            ]
            intro_body = "\n".join(intro_lines).strip()
            if intro_body:
                self._add_chunk(
                    ChildChunk(
                        chunk_id=f"{section_id}::intro",
                        section_id=section_id,
                        title=title,
                        text=f"[{section_id} {title}]\n{intro_body}",
                        is_table=False,
                    )
                )

        chunk_num = 0
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if not body and not heading:
                continue
            chunk_num += 1
            chunk_text = f"[{section_id} {title}] {heading}\n{body}".strip()
            self._add_chunk(
                ChildChunk(
                    chunk_id=f"{section_id}::h{chunk_num}",
                    section_id=section_id,
                    title=title,
                    text=chunk_text,
                    is_table=False,
                )
            )

    def _load_public_examples(self) -> None:
        pub_dir = self.settings.public_dir
        tutor_path = pub_dir / "tutor.jsonl"
        if tutor_path.exists():
            with tutor_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.tutor_examples.append(json.loads(line))

        guard_path = pub_dir / "guard.jsonl"
        if guard_path.exists():
            with guard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.guard_examples.append(json.loads(line))

    def _load_precomputed_assets(self) -> None:
        npz_path = self.settings.assets_dir / "kb_embeddings.npz"
        if not npz_path.exists():
            return
        try:
            data = np.load(npz_path, allow_pickle=False)
            if "chunk_embeddings" in data and "chunk_ids" in data:
                saved_ids = [str(x) for x in data["chunk_ids"].tolist()]
                saved_embs = np.asarray(data["chunk_embeddings"], dtype=np.float32)
                if len(saved_ids) == len(saved_embs) and saved_embs.ndim == 2:
                    id_to_row = {cid: idx for idx, cid in enumerate(saved_ids)}
                    aligned = np.zeros(
                        (len(self.chunks), saved_embs.shape[1]), dtype=np.float32
                    )
                    matched = 0
                    for idx, chunk in enumerate(self.chunks):
                        row_idx = id_to_row.get(chunk.chunk_id)
                        if row_idx is not None:
                            aligned[idx] = saved_embs[row_idx]
                            matched += 1
                    if matched == len(self.chunks) and matched > 0:
                        self._chunk_embeddings = _l2_normalize(aligned)
                    else:
                        logger.warning(
                            "Precomputed chunk embeddings mismatch (%d/%d matched); ignoring cache.",
                            matched,
                            len(self.chunks),
                        )

            if "contextual_chunk_ids" in data and "contextual_headers" in data:
                c_ids = [str(x) for x in data["contextual_chunk_ids"].tolist()]
                c_hdrs = [str(x) for x in data["contextual_headers"].tolist()]
                self._contextual_headers = dict(zip(c_ids, c_hdrs))

            if "tutor_embeddings" in data and len(data["tutor_embeddings"]) == len(
                self.tutor_examples
            ):
                self._tutor_embeddings = _l2_normalize(
                    np.asarray(data["tutor_embeddings"], dtype=np.float32)
                )
        except Exception as exc:
            logger.warning("Failed to load %s: %s", npz_path, exc)

    def _build_bm25_indices(self) -> None:
        chunk_corpus = [
            self._tokenize(
                f"{self._contextual_headers.get(c.chunk_id, '')} {c.text}".strip()
            )
            for c in self.chunks
        ]
        # Ensure no completely empty document breaks bm25s
        chunk_corpus = [toks if toks else ["_empty_"] for toks in chunk_corpus]
        self._chunk_bm25: Optional[bm25s.BM25] = None
        if chunk_corpus:
            self._chunk_bm25 = bm25s.BM25()
            self._chunk_bm25.index(chunk_corpus, show_progress=False)

        self._tutor_bm25: Optional[bm25s.BM25] = None
        if self.tutor_examples:
            tutor_corpus = []
            for ex in self.tutor_examples:
                user_text = " ".join(
                    m.get("content", "")
                    for m in ex.get("messages", [])
                    if m.get("role") == "user"
                )
                toks = self._tokenize(user_text)
                tutor_corpus.append(toks if toks else ["_empty_"])
            self._tutor_bm25 = bm25s.BM25()
            self._tutor_bm25.index(tutor_corpus, show_progress=False)

    async def ensure_embeddings(self) -> None:
        """Compute chunk and tutor embeddings on startup if cache is missing and embed_client is set."""
        if self.embed_client is None:
            return
        if self._chunk_embeddings is not None and (
            not self.tutor_examples or self._tutor_embeddings is not None
        ):
            return
        async with self._embed_lock:
            if self._chunk_embeddings is None and self.chunks:
                texts = [
                    f"{self._contextual_headers.get(c.chunk_id, '')}\n{c.text}".strip()
                    for c in self.chunks
                ]
                self._chunk_embeddings = await self._embed_texts(texts)
            if self._tutor_embeddings is None and self.tutor_examples:
                tutor_texts = [
                    " ".join(
                        m.get("content", "")
                        for m in ex.get("messages", [])
                        if m.get("role") == "user"
                    )
                    for ex in self.tutor_examples
                ]
                self._tutor_embeddings = await self._embed_texts(tutor_texts)

    async def _embed_texts(self, texts: list[str], batch_size: int = 128) -> Optional[np.ndarray]:
        if self.embed_client is None or not texts:
            return None
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = [normalize_text(t) or " " for t in texts[start : start + batch_size]]
                resp = await self.embed_client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch,
                )
                vectors.extend(item.embedding for item in resp.data)
            return _l2_normalize(np.asarray(vectors, dtype=np.float32))
        except Exception as exc:
            logger.warning("Embedding computation failed, falling back to BM25: %s", exc)
            return None

    async def _embed_query(self, query: str) -> Optional[np.ndarray]:
        if self.embed_client is None or not query.strip():
            return None
        try:
            resp = await self.embed_client.embeddings.create(
                model=self.settings.embedding_model,
                input=[collapse_korean_obfuscation(query)],
            )
            vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0.0 else vec
        except Exception as exc:
            logger.warning("Query embedding failed, using BM25 only: %s", exc)
            return None

    def _bm25_chunk_scores(self, query: str) -> np.ndarray:
        n_chunks = len(self.chunks)
        scores_arr = np.zeros(n_chunks, dtype=np.float32)
        if not n_chunks or self._chunk_bm25 is None:
            return scores_arr
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return scores_arr
        indices, scores = self._chunk_bm25.retrieve(
            [q_tokens], k=n_chunks, show_progress=False
        )
        for idx, sc in zip(indices[0], scores[0]):
            scores_arr[int(idx)] = float(sc)
        return scores_arr

    async def retrieve(
        self,
        query: str,
        top_k_sections: int = 4,
        prev_query: Optional[str] = None,
        last_weight: float = 0.75,
    ) -> list[RetrievedSection]:
        """Retrieve top_k_sections using Kiwi+BM25s + optional dense cosine via RRF (k=60)."""
        if not self.chunks or not self.sections:
            return []

        if self._chunk_embeddings is None and self.embed_client is not None:
            await self.ensure_embeddings()

        # Sparse BM25 scores (blended if prev_query is provided)
        bm25_scores = self._bm25_chunk_scores(query)
        if prev_query and prev_query.strip():
            prev_bm25 = self._bm25_chunk_scores(prev_query)
            bm25_scores = last_weight * bm25_scores + (1.0 - last_weight) * prev_bm25

        # Dense cosine scores if embeddings are available
        dense_scores = np.zeros(len(self.chunks), dtype=np.float32)
        has_dense = False
        if self._chunk_embeddings is not None and self.embed_client is not None:
            q_vec = await self._embed_query(query)
            if q_vec is not None:
                if prev_query and prev_query.strip():
                    prev_vec = await self._embed_query(prev_query)
                    if prev_vec is not None:
                        blended = last_weight * q_vec + (1.0 - last_weight) * prev_vec
                        norm = float(np.linalg.norm(blended))
                        if norm > 0.0:
                            q_vec = blended / norm
                dense_scores = self._chunk_embeddings @ q_vec
                has_dense = True

        # Reciprocal Rank Fusion (k=60) over chunks
        rrf_k = 60.0
        chunk_rrf = np.zeros(len(self.chunks), dtype=np.float32)

        bm25_order = np.argsort(-bm25_scores)
        for rank, idx in enumerate(bm25_order, start=1):
            if bm25_scores[idx] > 0.0:
                chunk_rrf[idx] += 1.0 / (rrf_k + rank)

        if has_dense:
            dense_order = np.argsort(-dense_scores)
            for rank, idx in enumerate(dense_order, start=1):
                chunk_rrf[idx] += 1.0 / (rrf_k + rank)

        # Aggregate chunk scores to parent sections
        sec_rrf: dict[str, float] = {}
        sec_bm25: dict[str, float] = {}
        sec_dense: dict[str, float] = {}
        for idx, chunk in enumerate(self.chunks):
            sid = chunk.section_id
            r_sc = float(chunk_rrf[idx])
            b_sc = float(bm25_scores[idx])
            d_sc = float(dense_scores[idx])
            if sid not in sec_rrf or r_sc > sec_rrf[sid]:
                sec_rrf[sid] = r_sc
            if sid not in sec_bm25 or b_sc > sec_bm25[sid]:
                sec_bm25[sid] = b_sc
            if sid not in sec_dense or d_sc > sec_dense[sid]:
                sec_dense[sid] = d_sc

        ranked_sids = sorted(
            sec_rrf.keys(),
            key=lambda s: (sec_rrf[s], sec_bm25[s], sec_dense[s]),
            reverse=True,
        )[:top_k_sections]

        return [
            RetrievedSection(
                section_id=sid,
                title=self.sections[sid].title,
                full_text=self.sections[sid].raw_text,
                rrf_score=sec_rrf[sid],
                bm25_score=sec_bm25[sid],
                dense_score=sec_dense[sid],
            )
            for sid in ranked_sids
        ]

    async def retrieve_similar_tutor_examples(
        self, query: str, k: int = 3
    ) -> list[dict[str, Any]]:
        """Return top-k similar public/tutor.jsonl exemplars via BM25/dense RRF."""
        n_ex = len(self.tutor_examples)
        if not n_ex or k <= 0:
            return []

        bm25_scores = np.zeros(n_ex, dtype=np.float32)
        if self._tutor_bm25 is not None:
            q_tokens = self._tokenize(query)
            if q_tokens:
                indices, scores = self._tutor_bm25.retrieve(
                    [q_tokens], k=n_ex, show_progress=False
                )
                for idx, sc in zip(indices[0], scores[0]):
                    bm25_scores[int(idx)] = float(sc)

        dense_scores = np.zeros(n_ex, dtype=np.float32)
        has_dense = False
        if self._tutor_embeddings is not None and self.embed_client is not None:
            q_vec = await self._embed_query(query)
            if q_vec is not None:
                dense_scores = self._tutor_embeddings @ q_vec
                has_dense = True

        rrf_k = 60.0
        rrf_scores = np.zeros(n_ex, dtype=np.float32)
        for rank, idx in enumerate(np.argsort(-bm25_scores), start=1):
            if bm25_scores[idx] > 0.0:
                rrf_scores[idx] += 1.0 / (rrf_k + rank)
        if has_dense:
            for rank, idx in enumerate(np.argsort(-dense_scores), start=1):
                rrf_scores[idx] += 1.0 / (rrf_k + rank)

        top_indices = np.argsort(-rrf_scores)[: min(k, n_ex)]
        return [self.tutor_examples[int(i)] for i in top_indices]

    def build_toc(self) -> str:
        """Build a compact 48-section directory for the static cached system prefix."""
        return "\n".join(
            f"- {sid}: {self.sections[sid].title}" for sid in self.section_order
        )
