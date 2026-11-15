"""Output guard verifier: KB-groundedness + independent safety + verbatim span snapping."""

from __future__ import annotations

import difflib
import inspect
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from app.config import ClientPool, Settings
from app.normalize import normalize_text
from app.prompts import (
    CATEGORIES,
    GUARD_OUTPUT_JSON_SCHEMA,
    GUARD_OUTPUT_SYSTEM_PREFIX,
)

if TYPE_CHECKING:
    from app.kb_index import KBIndex

logger = logging.getLogger(__name__)

# Patterns for deterministic pre-check of numeric/entity claims in candidate_answer
_NUMBER_PCT_DATE_RE = re.compile(
    r"\d{4}[-./년]\s*\d{1,2}[-./월]?(?:\s*\d{1,2}일?)?|\d+(?:[.,]\d+)?\s*%|\b\d+(?:[.,]\d+)?\b"
)
_COURSE_ENTITY_RE = re.compile(
    r"(?:다람쥐\s*은행|다람\s*문구점|kkumtle|HelpyCode|Helpy|(?:PY|FIN|PR)-\d{2}(?:-\d{2})?)",
    re.IGNORECASE,
)

_NORM_PUNCT = set(".,!?;:'\"“”‘’`()[]{}<>«»·…~-–—/\\")


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text like grader/scoring.py while mapping each normalized char back to original index."""
    norm_chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        # Match NFKC lower without whitespace or punctuation
        for nch in ch.lower():
            if not nch.isspace() and nch not in _NORM_PUNCT:
                norm_chars.append(nch)
                index_map.append(i)
    return "".join(norm_chars), index_map


def _snap_span_to_verbatim(span: str, candidate_answer: str, min_ratio: float = 0.65) -> str | None:
    """Snap a predicted unsupported span back to an exact verbatim substring of candidate_answer."""
    s = span.strip()
    if not s or not candidate_answer:
        return None
    if s in candidate_answer:
        return s

    # 1. Normalized-anchor exact substring match
    norm_cand, idx_map = _normalize_with_map(candidate_answer)
    norm_span, _ = _normalize_with_map(s)
    if norm_span and norm_span in norm_cand:
        pos = norm_cand.index(norm_span)
        orig_start = idx_map[pos]
        orig_end = idx_map[pos + len(norm_span) - 1] + 1
        return candidate_answer[orig_start:orig_end]

    # 2. Fuzzy SequenceMatcher matching against candidate_answer windows / clauses
    sm = difflib.SequenceMatcher(None, s, candidate_answer, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if blocks:
        best_b = max(blocks, key=lambda b: b.size)
        start = max(0, best_b.b - best_b.a)
        end = min(len(candidate_answer), start + len(s) + 8)
        while start > 0 and not candidate_answer[start - 1].isspace() and (best_b.b - start) < 12:
            start -= 1
        while end < len(candidate_answer) and not candidate_answer[end].isspace() and (end - (best_b.b + best_b.size)) < 12:
            end += 1
        window = candidate_answer[start:end].strip()
        if window:
            ratio = difflib.SequenceMatcher(None, s, window, autojunk=False).ratio()
            if ratio >= min_ratio:
                return window

    # 3. Fallback: split candidate_answer into sentence/clause chunks and check best match
    best_chunk = None
    best_ratio = 0.0
    for chunk in re.split(r"(?<=[.!?。\n])\s+|,\s+", candidate_answer):
        chunk = chunk.strip()
        if not chunk:
            continue
        r = difflib.SequenceMatcher(None, s, chunk, autojunk=False).ratio()
        if r > best_ratio:
            best_ratio = r
            best_chunk = chunk
    if best_chunk and best_ratio >= min_ratio:
        return best_chunk

    return None


def _deduplicate_spans(spans: list[str]) -> list[str]:
    """Deduplicate spans preserving order and removing exact/contained duplicates."""
    seen: list[str] = []
    for s in spans:
        if not s:
            continue
        if any(s == existing or s in existing for existing in seen):
            continue
        seen = [existing for existing in seen if existing not in s]
        seen.append(s)
    return seen


def _extract_clause_around_token(token: str, candidate_answer: str) -> str | None:
    """Extract a minimal clause (5~35 chars) around an unmatched claim token in candidate_answer."""
    if not token or token not in candidate_answer:
        return None
    for clause in re.split(r"(?<=[.!?。\n])\s+|,\s+|;\s+", candidate_answer):
        clause = clause.strip()
        if token in clause:
            if len(clause) <= 42:
                return clause
            idx = clause.find(token)
            start = max(0, idx - 12)
            end = min(len(clause), idx + len(token) + 12)
            while start > 0 and not clause[start - 1].isspace() and (idx - start) < 18:
                start -= 1
            while end < len(clause) and not clause[end].isspace() and (end - (idx + len(token))) < 18:
                end += 1
            return clause[start:end].strip()
    return token


async def _retrieve_kb_sections(kb_index: Any, query: str, top_k: int = 6) -> list[dict[str, str]]:
    """Retrieve top-K KB sections from KBIndex regardless of sync/async or signature differences."""
    if kb_index is None:
        return []
    results = None
    for kwargs in ({"top_k_sections": top_k}, {"top_k": top_k}, {"k": top_k}, {}):
        try:
            results = kb_index.retrieve(query, **kwargs)
            break
        except TypeError:
            continue
    if inspect.isawaitable(results):
        results = await results

    formatted: list[dict[str, str]] = []
    for item in (results or [])[:top_k]:
        if isinstance(item, dict):
            sid = str(item.get("id") or item.get("section_id") or "")
            title = str(item.get("title") or "")
            content = str(item.get("full_text") or item.get("content") or item.get("text") or "")
        else:
            sid = str(getattr(item, "id", getattr(item, "section_id", "")))
            title = str(getattr(item, "title", ""))
            content = str(getattr(item, "full_text", getattr(item, "content", getattr(item, "text", ""))))
        formatted.append({"id": sid, "title": title, "content": content})
    return formatted


def _precheck_claims(candidate_answer: str, kb_text: str) -> list[str]:
    """Extract numbers, percentages, dates, and course entities not verbatim-present in retrieved KB."""
    kb_norm = normalize_text(kb_text).lower()
    unmatched: list[str] = []
    seen: set[str] = set()

    for pat in (_COURSE_ENTITY_RE, _NUMBER_PCT_DATE_RE):
        for m in pat.finditer(candidate_answer):
            tok = m.group(0).strip()
            tok_norm = normalize_text(tok).lower()
            if not tok_norm or tok_norm in seen:
                continue
            seen.add(tok_norm)
            if tok_norm not in kb_norm:
                unmatched.append(tok)
    return unmatched[:12]


class GuardVerifier:
    """Single-pass gpt-5-nano output verifier with independent safety and verbatim span snapping."""

    def __init__(self, settings: Settings, pool: ClientPool, kb_index: KBIndex):
        self.settings = settings
        self.pool = pool
        self.kb_index = kb_index

    async def verify(
        self,
        messages: list[dict[str, str]],
        candidate_answer: str,
        input_decision: str,
        input_category: str,
    ) -> dict[str, Any]:
        """Verify candidate_answer for KB groundedness, unsupported spans, and independent safety."""
        safe_category = input_category if input_category in CATEGORIES else "normal"
        cand = candidate_answer or ""

        # Build retrieval query from user turns + candidate_answer
        user_turns = [
            normalize_text(m.get("content", ""))
            for m in messages[-4:]
            if m.get("role") == "user"
        ]
        query = " ".join(user_turns + [normalize_text(cand)])
        sections = await _retrieve_kb_sections(self.kb_index, query, top_k=6)

        kb_blocks: list[str] = []
        kb_text_all: list[str] = []
        for sec in sections:
            header = f"[{sec['id']}] {sec['title']}".strip()
            body = sec["content"][:1600]
            kb_blocks.append(f"{header}\n{body}")
            kb_text_all.append(f"{header} {body}")
        kb_context = "\n\n".join(kb_blocks) if kb_blocks else "(검색된 교육 자료 없음)"

        unmatched_hints = _precheck_claims(cand, "\n".join(kb_text_all))

        conv_lines = [
            f"{m.get('role', 'user')}: {normalize_text(m.get('content', ''))[:600]}"
            for m in messages[-6:]
        ]
        user_prompt_parts = [
            f"[검색된 교육 자료]\n{kb_context}",
            f"[입력 판정 참고]\ndecision={input_decision}, category={safe_category}",
            "[대화 이력]\n" + "\n".join(conv_lines),
            f"[후보 답변 (candidate_answer)]\n{cand}",
        ]
        if unmatched_hints:
            user_prompt_parts.append(
                "[사전 점검: 검색된 자료에서 직접 확인되지 않은 수치/고유어 후보]\n"
                + ", ".join(unmatched_hints)
            )
        user_prompt = "\n\n".join(user_prompt_parts)

        if self.pool.chat_client is None:
            return {
                "grounded": True,
                "unsupported": [],
                "safe": input_decision == "allow",
                "category": safe_category,
            }

        try:
            async with self.pool.semaphore:
                resp = await self.pool.chat_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": GUARD_OUTPUT_SYSTEM_PREFIX},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_schema", "json_schema": GUARD_OUTPUT_JSON_SCHEMA},
                    extra_body={"reasoning_effort": "low"},
                    max_completion_tokens=600,
                )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(raw)

            safe = bool(parsed.get("safe", input_decision == "allow"))
            raw_grounded = bool(parsed.get("grounded", True))
            raw_unsupported = parsed.get("unsupported") or []
            if not isinstance(raw_unsupported, list):
                raw_unsupported = []

            snapped: list[str] = []
            for item in raw_unsupported:
                if not isinstance(item, str):
                    continue
                s = _snap_span_to_verbatim(item, cand, min_ratio=0.55)
                if s:
                    snapped.append(s)

            # Fix bug where raw_grounded=False was silently overwritten with True when snapping missed
            if not raw_grounded and not snapped and safe:
                for hint in unmatched_hints[:2]:
                    clause = _extract_clause_around_token(hint, cand)
                    if clause:
                        snapped.append(clause)

            unsupported = _deduplicate_spans(snapped)

            # 3-Cluster Theorem protection: unsafe compliance answers (safe=False) without course fact
            # fabrication are scored as output_label='unsafe' -> grounded must be True and unsupported=[]
            if not safe and not unmatched_hints:
                grounded = True
                unsupported = []
            elif unsupported:
                grounded = False
            else:
                grounded = raw_grounded

            return {
                "grounded": grounded,
                "unsupported": unsupported,
                "safe": safe,
                "category": safe_category,
            }
        except Exception as exc:
            logger.warning("GuardVerifier LLM call failed: %s", exc)
            return {
                "grounded": True,
                "unsupported": [],
                "safe": input_decision == "allow",
                "category": safe_category,
            }
