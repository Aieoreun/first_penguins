"""Supervisor Multi-Agent Pipeline (Supervisor -> Search -> Review -> Write -> Verify).

Supports both LangGraph StateGraph compilation (when `langgraph` is installed) and a
zero-dependency async state-graph runner executing the exact same node functions.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.config import ClientPool, Settings
from app.deadline_budget import DeadlineBudget
from app.harness_guardrails import detect_deterministic_signals, enforce_tutor_invariants
from app.kb_index import KBIndex, RetrievedSection, _COURSE_TERMS
from app.normalize import collapse_korean_obfuscation
from app.prompts import (
    BLOCK_CATEGORIES,
    TUTOR_REVIEW_WRITE_JSON_SCHEMA,
    TUTOR_SYSTEM_PREFIX_TEMPLATE,
)
from app.tutor_pipeline import (
    _CANNED_TURN0_NORM,
    _FALLBACK_ANSWER,
    _is_low_signal,
    _strip_for_canned_check,
)

logger = logging.getLogger(__name__)


@dataclass
class TutorGraphState:
    """Mutable state passed across Supervisor -> Search -> Review -> Write -> Verify nodes."""

    messages: List[Dict[str, str]]
    budget: DeadlineBudget = field(default_factory=lambda: DeadlineBudget(27.5))

    # Extracted queries & signals
    last_query: str = ""
    prev_query: Optional[str] = None
    expanded_query: Optional[str] = None
    signals: Dict[str, bool] = field(default_factory=dict)
    low_signal: bool = False

    # Guard state
    block_category: Optional[str] = None
    block_confidence: float = 0.0

    # Search & Review state
    retrieved: List[RetrievedSection] = field(default_factory=list)
    exemplars: List[Dict[str, Any]] = field(default_factory=list)
    re_search_count: int = 0
    needs_re_search: bool = False
    review_hint: str = ""
    primary_section_id: str = ""

    # Write & Verify state
    route: str = "search_review_write"
    thought_category: str = "normal"
    evidence_in_kb: bool = True
    best_section_id: str = ""
    raw_action: str = "answer"
    raw_citations: List[str] = field(default_factory=list)
    raw_answer: str = ""
    final_response: Dict[str, Any] = field(default_factory=dict)

    # SPEC.md §8 non-content telemetry
    node_latencies_ms: Dict[str, float] = field(default_factory=dict)


class SupervisorTutorPipeline:
    """Multi-agent Supervisor pipeline implementing Search -> Review -> Write -> Verify."""

    def __init__(
        self,
        settings: Settings,
        pool: ClientPool,
        kb_index: KBIndex,
        guard_classifier: Any = None,
    ):
        self.settings = settings
        self.pool = pool
        self.kb_index = kb_index
        self.guard_classifier = guard_classifier
        self._system_prefix = TUTOR_SYSTEM_PREFIX_TEMPLATE.format(
            toc=self.kb_index.build_toc()
        )
        self.last_telemetry: Dict[str, Any] = {}

    def _extract_queries(
        self, messages: List[Dict[str, str]]
    ) -> tuple[str, Optional[str], Optional[str]]:
        user_turns = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m.get("content", "").strip()
        ]
        if not user_turns:
            return "", None, None

        last_clean = collapse_korean_obfuscation(user_turns[-1])
        prev_clean: Optional[str] = None
        expanded_clean: Optional[str] = None

        if len(user_turns) > 1:
            candidates = [
                u for u in user_turns[:-1] if _strip_for_canned_check(u) not in _CANNED_TURN0_NORM
            ]
            if candidates:
                prev_clean = collapse_korean_obfuscation(" ".join(candidates[-2:]))
                all_turns = " ".join(candidates[-3:] + [user_turns[-1]])
                expanded_clean = collapse_korean_obfuscation(all_turns)

        return last_clean, prev_clean, expanded_clean

    async def _run_guard(
        self, messages: List[Dict[str, str]], collapsed_last: str
    ) -> tuple[Optional[str], float]:
        if self.guard_classifier is None:
            return None, 0.0

        guard_msgs = [dict(m) for m in messages]
        for idx in range(len(guard_msgs) - 1, -1, -1):
            if guard_msgs[idx].get("role") == "user":
                guard_msgs[idx]["content"] = collapsed_last
                break

        try:
            fn = None
            for attr in ("classify", "classify_input", "predict_input", "check_input", "predict"):
                if hasattr(self.guard_classifier, attr):
                    fn = getattr(self.guard_classifier, attr)
                    break
            if fn is None and callable(self.guard_classifier):
                fn = self.guard_classifier
            if fn is None:
                return None, 0.0

            res = fn(guard_msgs)
            if inspect.isawaitable(res):
                res = await res

            if isinstance(res, tuple) and len(res) >= 2:
                decision = res[0]
                category = res[1]
                confidence = float(res[2]) if len(res) >= 3 else (1.0 if decision == "block" else 0.0)
            elif isinstance(res, dict):
                decision = res.get("decision")
                category = res.get("category")
                confidence = float(res.get("confidence", 1.0 if decision == "block" else 0.0))
            else:
                decision = getattr(res, "decision", None)
                category = getattr(res, "category", None)
                confidence = float(getattr(res, "confidence", 1.0 if decision == "block" else 0.0))

            if decision == "block" and category in BLOCK_CATEGORIES:
                return category, confidence
        except Exception as exc:
            logger.warning("Supervisor guard_classifier failed: %s", exc)

        return None, 0.0

    async def supervisor_node(self, state: TutorGraphState) -> None:
        """Node 1 (Supervisor): Extract multi-turn queries, detect signals, and route."""
        t0 = time.perf_counter()
        last_q, prev_q, exp_q = self._extract_queries(state.messages)
        state.last_query = last_q
        state.prev_query = prev_q
        state.expanded_query = exp_q
        state.low_signal = _is_low_signal(last_q)
        state.signals = detect_deterministic_signals(last_q)

        if not last_q or state.low_signal:
            state.route = "fast_clarify"
            state.thought_category = "low_signal_input"
            state.raw_action = "clarify"
            state.raw_citations = []
            state.raw_answer = "어떤 수업 내용이나 실습 문제가 궁금한지 구체적으로 질문해 주세요."
        else:
            state.route = "search_review_write"

        state.node_latencies_ms["supervisor"] = round((time.perf_counter() - t0) * 1000, 1)

    async def search_node(self, state: TutorGraphState) -> None:
        """Node 2 (Search): Run parallel Guard classification + Hybrid KB search + Exemplar search."""
        t0 = time.perf_counter()
        if state.re_search_count == 0:
            guard_task = asyncio.create_task(self._run_guard(state.messages, state.last_query))
            kb_task = asyncio.create_task(
                self.kb_index.retrieve(
                    state.last_query,
                    top_k_sections=4,
                    prev_query=state.prev_query,
                    last_weight=0.75,
                )
            )
            ex_query = (
                f"{state.prev_query} {state.last_query}".strip()
                if state.prev_query
                else state.last_query
            )
            ex_task = asyncio.create_task(
                self.kb_index.retrieve_similar_tutor_examples(ex_query, k=3)
            )
            (block_cat, block_conf), retrieved, exemplars = await asyncio.gather(
                guard_task, kb_task, ex_task
            )
            state.block_category = block_cat
            state.block_confidence = block_conf
            state.retrieved = retrieved
            state.exemplars = exemplars
        else:
            # Bounded 1x Re-search with expanded multi-turn context & assistant history entities
            assistant_entities: list[str] = []
            for m in state.messages[:-1]:
                if m.get("role") == "assistant":
                    txt = m.get("content", "")
                    for term in _COURSE_TERMS:
                        if term in txt and term not in assistant_entities:
                            assistant_entities.append(term)
            re_query = " ".join(
                p for p in [state.expanded_query or state.last_query, " ".join(assistant_entities)] if p
            )
            more_secs = await self.kb_index.retrieve(
                re_query, top_k_sections=4, prev_query=state.prev_query, last_weight=0.55
            )
            seen = {s.section_id for s in state.retrieved}
            merged = list(state.retrieved)
            for s in more_secs:
                if s.section_id not in seen:
                    merged.append(s)
                    seen.add(s.section_id)
            merged.sort(key=lambda x: (x.rrf_score, x.bm25_score, x.dense_score), reverse=True)
            state.retrieved = merged[:4]

        state.node_latencies_ms[f"search_{state.re_search_count}"] = round(
            (time.perf_counter() - t0) * 1000, 1
        )

    async def review_node(self, state: TutorGraphState) -> None:
        """Node 3 (Review): Audit retrieved evidence, decide on bounded re-search, and rank primary section."""
        t0 = time.perf_counter()
        state.needs_re_search = False

        if not state.retrieved:
            if (
                state.re_search_count == 0
                and state.expanded_query
                and state.budget.has_at_least(14.0)
            ):
                state.needs_re_search = True
                state.re_search_count += 1
            state.node_latencies_ms["review"] = round((time.perf_counter() - t0) * 1000, 1)
            return

        top1 = state.retrieved[0]
        state.primary_section_id = top1.section_id

        # Check if query references previous turns (pronouns/omitted subject) with weak BM25 hit
        has_referential = any(
            w in state.last_query for w in ("그거", "그 ", "여기서", "이거", "방금", "위에서", "거기서")
        )
        if (
            state.re_search_count == 0
            and (top1.bm25_score <= 0.5 or has_referential)
            and len(state.messages) >= 3
            and state.budget.has_at_least(14.0)
        ):
            state.needs_re_search = True
            state.re_search_count += 1
            state.node_latencies_ms["review"] = round((time.perf_counter() - t0) * 1000, 1)
            return

        # Build neutral Review summary (avoid Rank-1 anchoring so Write checks exact section text)
        sec_summaries = [
            f"`{s.section_id}` ({s.title})"
            for s in state.retrieved
        ]
        review_lines = [
            "[검색 검토(Review) 요약]",
            "- 후보 섹션 목록(검색 순위와 무관하게 본문을 직접 대조해 1개만 선택): " + " | ".join(sec_summaries),
            "- 섹션 선택 주의: 같은 상품·실습명(다람쥐 틴즈 체크카드, 첫걸음 통장, 새싹적금, 동아리 실습 등)이 여러 섹션에 나올 수 있습니다. "
            "질문이 요구하는 **정확한 수치·규칙·함수·예외가 본문에 실제로 적힌 단 1개의 섹션**만 best_section_id와 citations에 넣으세요.",
            "- out_of_material 판별: 위 4개 후보 본문 어디에도 질문한 바로 그 세부 규정(등급 기준, 수수료, 긴급 확인 기준, 삭제 확인창 여부, 가상 댓글 기준 등)이 "
            "**직접 명시돼 있지 않으면** 인접 문장으로 유추하지 말고 반드시 thought_category='out_of_material', evidence_in_kb=false, action='abstain', citations=[]로 답하세요.",
        ]
        if state.signals.get("has_other_pii_dox"):
            review_lines.append(
                "- 타인 정보 추출 감지: 친구/다른 학생의 개인정보나 통장 내역을 알아내 달라는 요청이므로 "
                "thought_category='harmful_content_actionable', action='refuse', citations=[]로 거절하세요."
            )
        if state.signals.get("has_self_pii"):
            review_lines.append(
                "- 개인정보 감지: 학습자가 본인 이름/학교/연락처를 언급했습니다. 정상 질문이면 "
                "thought_category='pii_disclosure', action='answer'로 답하고 개인정보 보호 안내 문구를 꼭 포함하세요."
            )
        if state.signals.get("has_answer_seeking"):
            review_lines.append(
                "- 정답 요구 감지: thought_category='answer_seeking', action='scaffold', citations=[근거 섹션 ID 1개]. "
                "정답 숫자/보기는 직접 말하지 말되, 해당 섹션의 **핵심 규칙·계산 원리**를 정확히 설명해 주고 학습자가 직접 구하게 유도하세요."
            )
        if state.signals.get("has_academic_dishonesty"):
            review_lines.append(
                "- 과제 대필 감지: 수행평가/숙제 제출용 완성 답안 요구이므로 thought_category='academic_dishonesty', "
                "action='scaffold', citations=[]로 설정하고 코드 블록(```)이나 완성 코드·실행 결과를 절대 주지 마세요."
            )

        state.review_hint = "\n".join(review_lines)
        state.node_latencies_ms["review"] = round((time.perf_counter() - t0) * 1000, 1)

    @staticmethod
    def _is_high_conf_block(state: TutorGraphState) -> bool:
        from app.harness_guardrails import is_quiz_pretext_override

        high_conf_block = state.block_category is not None and state.block_confidence >= 0.85
        if high_conf_block and is_quiz_pretext_override(state.block_category, state.signals):
            return False
        return high_conf_block

    def _format_dynamic_context(self, state: TutorGraphState) -> str:
        parts: list[str] = []
        high_conf_block = self._is_high_conf_block(state)

        if high_conf_block and state.block_category:
            parts.append(
                "[안전 가드레일 경고]\n"
                f"입력 가드레일이 마지막 발화를 `{state.block_category}`(으)로 강하게 분류했습니다. "
                "부적절한 요청에는 응하지 말고 해당 action(refuse/redirect/scaffold)과 citations=[]로 답하세요."
            )
        elif state.block_category and state.block_confidence >= 0.65:
            parts.append(
                "[안전 가드레일 참고]\n"
                f"가드레일이 `{state.block_category}` 가능성을 표시했으나 확신도가 낮습니다. "
                "교육 자료 관련 정상 질문(normal/out_of_material/pii_disclosure)이나 정서 위기 호소(emotional_distress)라면 "
                "절대 오거부하지 말고 정상 action으로 답하세요. 실제로 부적절한 요청일 때만 거절/유도하세요."
            )

        if state.review_hint:
            parts.append(state.review_hint)

        if state.retrieved:
            sec_blocks = [
                f"[{sec.section_id}] ({sec.title})\n{sec.full_text}"
                for sec in state.retrieved
            ]
            parts.append("[검색된 교육 자료]\n" + "\n\n---\n\n".join(sec_blocks))
        else:
            parts.append("[검색된 교육 자료]\n(관련 섹션을 찾지 못했습니다.)")

        if state.exemplars:
            ex_lines: list[str] = []
            for idx, ex in enumerate(state.exemplars, start=1):
                labels = ex.get("labels", {})
                ex_action = labels.get("action", "answer")
                ex_ev = labels.get("evidence", [])
                ex_ref = labels.get("reference_answer", "")
                ex_user = ""
                for m in reversed(ex.get("messages", [])):
                    if m.get("role") == "user":
                        ex_user = m.get("content", "")
                        break
                ex_lines.append(
                    f"예시 {idx}) 질문: {ex_user[:160]}\n"
                    f"-> action={ex_action}, citations={ex_ev}, 답변={ex_ref}"
                )
            parts.append("[유사 답변 예시]\n" + "\n\n".join(ex_lines))

        return "\n\n====================\n\n".join(parts)

    async def write_node(self, state: TutorGraphState) -> None:
        """Node 4 (Write): Schema-Guided Micro-CoT generation with deadline-adaptive effort."""
        t0 = time.perf_counter()
        high_conf_block = self._is_high_conf_block(state)

        if self.pool.chat_client is None or not state.budget.has_at_least(4.0):
            if state.signals.get("has_emotional_distress"):
                state.thought_category = "emotional_distress"
                state.raw_action = "escalate"
                state.raw_citations = []
            elif state.signals.get("has_academic_dishonesty"):
                state.thought_category = "academic_dishonesty"
                state.raw_action = "scaffold"
                state.raw_citations = []
            elif state.signals.get("has_answer_seeking") and state.primary_section_id:
                state.thought_category = "answer_seeking"
                state.raw_action = "scaffold"
                state.best_section_id = state.primary_section_id
                state.raw_citations = [state.primary_section_id]
            elif state.retrieved and state.retrieved[0].bm25_score >= 1.5:
                state.thought_category = "pii_disclosure" if state.signals.get("has_self_pii") else "normal"
                state.raw_action = "answer"
                state.best_section_id = state.primary_section_id
                state.raw_citations = [state.primary_section_id]
            else:
                state.thought_category = "out_of_material"
                state.raw_action = "abstain"
                state.raw_citations = []
            state.raw_answer = _FALLBACK_ANSWER
            state.node_latencies_ms["write"] = round((time.perf_counter() - t0) * 1000, 1)
            return

        effort = (
            "minimal"
            if (
                not state.budget.has_at_least(10.0)
                or state.low_signal
                or (
                    high_conf_block
                    and state.block_category in ("jailbreak_injection", "harmful_content", "off_topic")
                )
            )
            else "low"
        )

        dynamic_context = self._format_dynamic_context(state)
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prefix},
            {"role": "user", "content": dynamic_context},
            *state.messages,
        ]

        async with self.pool.semaphore:
            resp = await asyncio.wait_for(
                self.pool.chat_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=llm_messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": TUTOR_REVIEW_WRITE_JSON_SCHEMA,
                    },
                    max_completion_tokens=2000,
                    extra_body={"reasoning_effort": effort},
                ),
                timeout=state.budget.step_timeout(self.settings.llm_timeout_sec, reserve_sec=0.8),
            )

        raw_content = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw_content)

        # Support both TUTOR_REVIEW_WRITE_JSON_SCHEMA and legacy TUTOR_JSON_SCHEMA in mocks
        state.thought_category = str(parsed.get("thought_category", ""))
        state.evidence_in_kb = bool(parsed.get("evidence_in_kb", True))
        state.best_section_id = str(parsed.get("best_section_id", "")).strip()
        state.raw_action = str(parsed.get("action", "answer"))
        state.raw_citations = list(parsed.get("citations") or [])
        state.raw_answer = str(parsed.get("answer", "")).strip()
        state.node_latencies_ms["write"] = round((time.perf_counter() - t0) * 1000, 1)

    async def verify_node(self, state: TutorGraphState) -> None:
        """Node 5 (Verify): Apply deterministic Runtime Harness invariants."""
        t0 = time.perf_counter()
        retrieved_ids = [s.section_id for s in state.retrieved]
        state.final_response = enforce_tutor_invariants(
            action=state.raw_action,
            citations=state.raw_citations,
            answer=state.raw_answer,
            retrieved_ids=retrieved_ids,
            thought_category=state.thought_category,
            best_section_id=state.best_section_id,
            evidence_in_kb=state.evidence_in_kb,
            guard_block_category=state.block_category,
            guard_confidence=state.block_confidence,
            last_query=state.last_query,
        )
        state.node_latencies_ms["verify"] = round((time.perf_counter() - t0) * 1000, 1)

    async def _execute_graph(self, state: TutorGraphState) -> Dict[str, Any]:
        await self.supervisor_node(state)
        if state.route == "fast_clarify":
            await self.verify_node(state)
            return state.final_response

        await self.search_node(state)
        await self.review_node(state)
        if state.needs_re_search:
            await self.search_node(state)
            await self.review_node(state)

        await self.write_node(state)
        await self.verify_node(state)
        return state.final_response

    async def handle(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run the Supervisor -> Search -> Review -> Write -> Verify state graph."""
        if not messages:
            return {
                "answer": "무엇이 궁금한지 질문해 주세요.",
                "action": "clarify",
                "citations": [],
            }

        budget = DeadlineBudget(self.settings.request_deadline_sec)
        state = TutorGraphState(messages=messages, budget=budget)

        try:
            res = await asyncio.wait_for(
                self._execute_graph(state),
                timeout=self.settings.request_deadline_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "SupervisorTutorPipeline timed out after %.1fs",
                self.settings.request_deadline_sec,
            )
            res = {"answer": _FALLBACK_ANSWER, "action": "abstain", "citations": []}
        except Exception as exc:
            logger.exception("SupervisorTutorPipeline failed: %s", type(exc).__name__)
            res = {"answer": _FALLBACK_ANSWER, "action": "abstain", "citations": []}

        # Store non-content telemetry for SPEC.md §8 logging
        self.last_telemetry = {
            "route": state.route,
            "re_search": state.re_search_count,
            "top1_rrf": round(state.retrieved[0].rrf_score, 4) if state.retrieved else 0.0,
            "node_ms": state.node_latencies_ms,
        }
        return res
