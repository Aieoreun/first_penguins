"""요청 하나를 처리하는 흐름. docs/sw/02_설계.md 결정 9·10·13·14·15.

/tutor: 검색 → 입력 가드(분류 + 근거 섹션) → 고정 문구 또는 답변 생성
/guard: 검색 → 입력 가드 → (후보 답변이 있으면) 출력 가드

모델 호출은 요청당 두 번까지다. 어느 단계가 실패해도 규격에 맞는 응답을 낸다.
state.errors에는 단계 이름과 오류 종류만 적는다. 요청·답변 본문은 적지 않는다.
"""
import asyncio
import json
import os
import re
from typing import List, Optional

from . import prompts
from .schema import Decision, Label, parse_label
from .search import Searcher
from .spans import locate_all
from .state import Chunk, State

FALLBACK_ANSWER = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."

# 요청당 30초 안에 끝나도록 단계마다 제한 시간을 둔다(초). 합이 30보다 작아야 한다
EMBED_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", "3"))
CLASSIFY_TIMEOUT = float(os.environ.get("CLASSIFY_TIMEOUT", "10"))
SECOND_TIMEOUT = float(os.environ.get("SECOND_TIMEOUT", "13"))

# 추론 수준. 점수를 보고 고른다 (결정 15, 미확인)
CLASSIFY_EFFORT = os.environ.get("CLASSIFY_EFFORT", "low")
ANSWER_EFFORT = os.environ.get("ANSWER_EFFORT", "minimal")
GUARD_EFFORT = os.environ.get("GUARD_EFFORT", "low")

QUERY_MAX_CHARS = 6000        # 임베딩에 넘기는 검색어 길이 한도
FALLBACK_CHUNKS = 5           # 근거 섹션이 없을 때 출력 가드에 넘기는 조각 수

# 근거 섹션을 인용하는 분류 (결정 14). out_of_material은 가까운 내용을 덧붙였을 때만 넣는다
_CITE = {Label.NORMAL, Label.PII_DISCLOSURE, Label.ANSWER_SEEKING, Label.ACADEMIC_DISHONESTY}
# 답변 끝에 섹션 ID를 붙이는 분류 (결정 12-1). 공개 세트에서 정답 근거가 있는 범주다
_APPEND_ID = {Label.NORMAL, Label.PII_DISCLOSURE, Label.ANSWER_SEEKING}
# 근거 섹션이 꼭 있어야 하는 분류. 입력 가드가 못 고르면 검색 1위 섹션을 쓴다
_NEEDS_SECTION = _APPEND_ID

_SECTION_MARK = re.compile(r"\s*[\(\[][A-Z]{2,4}-\d{2}-\d{2}[\)\]]")
_SENTENCE_END = re.compile(r"[.!?。](\s|$)")


def _json_object(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


class Pipeline:
    def __init__(self, searcher: Searcher, llm):
        self.searcher = searcher
        self.llm = llm

    # ---------- 단계 ----------

    async def _search(self, state: State, extra: str = "") -> None:
        query = "\n".join(m["content"] for m in state.messages)
        if extra:
            query += "\n" + extra
        vector = None
        if self.searcher.has_embeddings and self.llm.can_embed:
            try:
                vectors = await self.llm.embed([query[:QUERY_MAX_CHARS]], EMBED_TIMEOUT)
                vector = vectors[0] if vectors else None
            except Exception as e:  # 임베딩 실패 → 낱말 검색만 (결정 7)
                state.errors.append(f"embed:{type(e).__name__}")
        found = self.searcher.search(query, vector)
        state.chunks = [Chunk(c.section_id, c.text) for c in found]

    async def _classify(self, state: State) -> None:
        candidates = {c.section_id for c in state.chunks}
        try:
            text = await self.llm.chat(
                prompts.CLASSIFY_SYSTEM,
                prompts.classify_user(state.chunks, state.history, state.last_user),
                effort=CLASSIFY_EFFORT, max_tokens=1500, timeout=CLASSIFY_TIMEOUT, json_mode=True)
            state.label = parse_label(text)
            if state.label is None:
                state.errors.append("classify:parse")
            obj = _json_object(text) or {}
            section = obj.get("section")
            if isinstance(section, str) and section.strip() in candidates:
                state.section = section.strip()  # 검색 후보에 없는 ID는 버린다
        except Exception as e:  # 분류 실패 → normal (결정 15)
            state.errors.append(f"classify:{type(e).__name__}")
        label = state.label if state.label is not None else Label.NORMAL
        if state.section is None and label in _NEEDS_SECTION and state.chunks:
            state.section = state.chunks[0].section_id

    async def _generate(self, state: State) -> None:
        label = state.label if state.label is not None else Label.NORMAL
        material = self.searcher.section_text(state.section) if state.section else ""
        try:
            text = await self.llm.chat(
                prompts.ANSWER_SYSTEMS[label],
                prompts.answer_user(material, state.history, state.last_user),
                effort=ANSWER_EFFORT, max_tokens=2000, timeout=SECOND_TIMEOUT)
        except Exception as e:  # 답변 생성 실패 → 사과 문구, 인용 없음 (결정 15)
            state.errors.append(f"generate:{type(e).__name__}")
            text = ""
        text = _SECTION_MARK.sub("", text).strip()  # 모델이 적은 섹션 ID는 지운다. 코드가 붙인다
        if not text:
            state.answer, state.citations = FALLBACK_ANSWER, []
            return
        cite = bool(state.section) and (
            label in _CITE
            or (label is Label.OUT_OF_MATERIAL and len(_SENTENCE_END.findall(text)) >= 2))
        if cite and label in _APPEND_ID:
            text = f"{text} ({state.section})"
        state.answer = text
        state.citations = [state.section] if cite else []

    async def _check_output(self, state: State) -> None:
        candidate = state.candidate_answer or ""
        if state.decision is Decision.BLOCK:
            # 차단된 입력: 후보 답변이 요청을 들어줬는지만 본다. 근거 검사는 하지 않는다 (결정 10)
            state.grounded, state.unsupported = True, []
            try:
                text = await self.llm.chat(
                    prompts.COMPLIED_SYSTEM,
                    prompts.complied_user(state.category.value, state.history, state.last_user, candidate),
                    effort=GUARD_EFFORT, max_tokens=1500, timeout=SECOND_TIMEOUT, json_mode=True)
                complied = (_json_object(text) or {}).get("complied")
                state.safe = complied is False  # 판정을 못 읽으면 안전하지 않다고 본다
                if not isinstance(complied, bool):
                    state.errors.append("complied:parse")
            except Exception as e:
                state.errors.append(f"complied:{type(e).__name__}")
                state.safe = False
            return
        # 허용된 입력: 근거 없는 구절을 찾는다. safe는 규칙으로 true다 (결정 10)
        state.safe = True
        if state.section:
            material = self.searcher.section_text(state.section)
        else:
            material = "\n\n".join(c.text for c in state.chunks[:FALLBACK_CHUNKS])
        try:
            text = await self.llm.chat(
                prompts.UNSUPPORTED_SYSTEM, prompts.unsupported_user(material, candidate),
                effort=GUARD_EFFORT, max_tokens=1500, timeout=SECOND_TIMEOUT, json_mode=True)
            phrases = (_json_object(text) or {}).get("unsupported")
            if not isinstance(phrases, list):
                state.errors.append("unsupported:parse")
                phrases = []
            state.unsupported = locate_all(candidate, phrases)  # 후보 답변의 글자 그대로 (결정 11)
        except Exception as e:  # 근거 검사 실패 → 근거 있음 (결정 15)
            state.errors.append(f"unsupported:{type(e).__name__}")
            state.unsupported = []
        state.grounded = not state.unsupported

    # ---------- 요청 ----------

    async def tutor(self, body) -> State:
        state = State.from_request(body)
        await self._search(state)
        await self._classify(state)
        if state.needs_generation:
            await self._generate(state)
        return state

    async def guard(self, body) -> State:
        state = State.from_request(body)
        await self._search(state, extra=state.candidate_answer or "")
        await self._classify(state)
        if state.has_candidate:
            await self._check_output(state)
        return state
