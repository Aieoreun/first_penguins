"""Dual-Perspective & Tie-Breaking Supervisor Agent for /guard (Input + Output)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from app.config import ClientPool, Settings
from app.deadline_budget import DeadlineBudget
from app.guard_classifier import (
    _DIRECT_ATTACK_OVERRIDE_RE,
    _META_DEFENSE_EDUCATIONAL_RE,
    GuardClassifier,
    _format_messages_for_llm,
)
from app.guard_verifier import (
    GuardVerifier,
    _deduplicate_spans,
    _precheck_claims,
    _retrieve_kb_sections,
    _snap_span_to_verbatim,
)
from app.normalize import normalize_text
from app.prompts import (
    BLOCK_CATEGORIES,
    CATEGORIES,
    GUARD_INPUT_TIEBREAKER_SCHEMA,
    GUARD_INPUT_TIEBREAKER_SYSTEM_PREFIX,
    GUARD_OUTPUT_GROUNDING_SCHEMA,
    GUARD_OUTPUT_GROUNDING_SYSTEM_PREFIX,
    GUARD_OUTPUT_JUDGE_SCHEMA,
    GUARD_OUTPUT_JUDGE_SYSTEM_PREFIX,
    GUARD_OUTPUT_SAFETY_SCHEMA,
    GUARD_OUTPUT_SAFETY_SYSTEM_PREFIX,
)

logger = logging.getLogger("first_penguins.guard_supervisor")

# Linguistic domain cues (used only to gate when 2nd-perspective tie-breaking adds value)
_COURSE_DOMAIN_CUE_RE = re.compile(
    r"(?:다람쥐|체크카드|틴즈|통장|새싹\s*적금|꿈모으기|정기예금|도토리|다람\s*문구점|문구점|"
    r"파이썬|프롬프트|울타리|거름망|실습|교재|학습자료|수업|동아리|kkumtle|helpy|fruit_|"
    r"\bround\b|\bint\b|\bif\b|\bwhile\b|\bfor\b|\bdef\b|\bpass\b|\bcontinue\b|\bbreak\b)",
    re.IGNORECASE,
)
_OOM_DETAIL_CUE_RE = re.compile(
    r"(?:수수료|한도|등급|확인창|특별\s*해지|재발급|유효\s*기간|가입\s*대상|가입\s*나이|만\s*\d+세|"
    r"이자율|금리|연\s*\d|몇\s*(?:번|개|회|명|일|시간|세|살|원)|얼마|무조건|차단|포함되|기준|예외|환불|결제|적립|할인)"
)
_CODE_OR_EVAL_CUE_RE = re.compile(
    r"(?:코드|수행평가|과제|숙제|제출|답안|정답|보기|몇\s*번|역할|울타리|지시|프롬프트)"
)


class GuardSupervisor:
    """Supervisor running concurrent /guard Input + Output multi-perspective validation."""

    def __init__(
        self,
        settings: Settings,
        pool: ClientPool,
        kb_index: Any,
        guard_classifier: GuardClassifier,
        guard_verifier: Optional[GuardVerifier] = None,
    ):
        self.settings = settings
        self.pool = pool
        self.kb_index = kb_index
        self.guard_classifier = guard_classifier
        self.guard_verifier = guard_verifier

    def _select_tiebreak_pair(
        self,
        decision: str,
        category: str,
        conf: float,
        diag: dict[str, Any],
    ) -> Optional[tuple[str, str]]:
        """Gate 2nd-perspective input tie-breaker strictly on uncertain or boundary cases."""
        if diag.get("deterministic") or conf >= 0.96:
            return None

        last_user = str(diag.get("last_user") or "")
        signals = diag.get("signals") or {}
        max_bm25 = float(diag.get("max_bm25") or 0.0)
        cpu_cat = str(diag.get("cpu_category") or "normal")
        top2_cats = list(diag.get("top2_cats") or [])

        # Deterministic safety invariants stay locked
        if signals.get("has_other_pii_dox") or signals.get("has_hate_insult") or signals.get("has_emotional_distress"):
            return None

        # 1. out_of_material vs normal / off_topic boundary (strictly when weak KB hit)
        if category == "out_of_material":
            return ("out_of_material", "normal")
        if (
            category in ("normal", "off_topic")
            and _COURSE_DOMAIN_CUE_RE.search(last_user)
            and max_bm25 < 0.7
        ):
            return (category, "out_of_material")

        # 2. CPU TF-IDF vs Primary LLM disagreement
        if conf < 0.85 and not diag.get("is_meta_normal"):
            alt = cpu_cat if cpu_cat != category else ("normal" if decision == "block" else "off_topic")
            return (category, alt)

        # 3. academic_dishonesty / answer_seeking vs normal boundary
        if category in ("academic_dishonesty", "answer_seeking") and not (
            signals.get("has_academic_dishonesty") or signals.get("has_answer_seeking")
        ):
            if any(k in last_user for k in ("정답", "퀴즈", "문제", "몇 번", "보기")):
                alt = "answer_seeking" if category == "academic_dishonesty" else "academic_dishonesty"
                return (category, alt)
            return (category, "normal")

        if (
            category == "normal"
            and not signals.get("has_self_pii")
            and not diag.get("is_meta_normal")
            and _CODE_OR_EVAL_CUE_RE.search(last_user)
            and len(top2_cats) > 1
            and top2_cats[1] in ("academic_dishonesty", "answer_seeking", "jailbreak_injection")
            and float(diag.get("cat_margin") or 1.0) < 0.35
        ):
            return ("normal", top2_cats[1])

        # 4. Non-imperative jailbreak vs normal
        if category == "jailbreak_injection" and not diag.get("jb_boost") and conf < 0.95:
            return ("jailbreak_injection", "normal")

        return None

    async def _run_input_tiebreaker(
        self,
        messages: list[dict[str, str]],
        cand_a: str,
        cand_b: str,
        diag: dict[str, Any],
        budget: DeadlineBudget,
    ) -> Optional[tuple[str, str]]:
        """Run pairwise contrastive 2nd-perspective auditor over top-2 competing categories."""
        if self.pool.chat_client is None or not budget.has_at_least(8.0):
            return None

        parts: list[str] = [
            f"[경합 후보 범주]\n- 후보 A: {cand_a}\n- 후보 B: {cand_b}\n(오직 위 두 후보 중 실제 의도와 교재 근거에 더 정확히 부합하는 1개만 선택하세요.)"
        ]
        kb_summary = str(diag.get("kb_summary") or "").strip()
        if kb_summary:
            parts.append(f"[교재 검색 결과]\n{kb_summary}")
        parts.append("[현재 대화 이력]\n" + _format_messages_for_llm(messages))
        user_prompt = "\n\n".join(parts)

        efforts: tuple[tuple[str, int], ...] = (
            (("low", 1000), ("minimal", 800))
            if budget.has_at_least(14.0)
            else (("minimal", 800),)
        )
        for effort, tok_cap in efforts:
            try:
                async with self.pool.semaphore:
                    resp = await asyncio.wait_for(
                        self.pool.chat_client.chat.completions.create(
                            model=self.settings.openai_model,
                            messages=[
                                {"role": "system", "content": GUARD_INPUT_TIEBREAKER_SYSTEM_PREFIX},
                                {"role": "user", "content": user_prompt},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": GUARD_INPUT_TIEBREAKER_SCHEMA,
                            },
                            extra_body={"reasoning_effort": effort},
                            max_completion_tokens=tok_cap,
                        ),
                        timeout=budget.step_timeout(self.settings.llm_timeout_sec, reserve_sec=1.5),
                    )
                raw = (resp.choices[0].message.content or "").strip()
                if not raw:
                    continue
                parsed = json.loads(raw)
                cat = str(parsed.get("category") or "")
                if cat not in CATEGORIES or cat not in (cand_a, cand_b):
                    return None
                dec = "block" if cat in BLOCK_CATEGORIES else "allow"
                return dec, cat
            except Exception as exc:
                logger.warning("GuardSupervisor tiebreaker attempt failed effort=%s err=%s", effort, type(exc).__name__)
                if not budget.has_at_least(4.5):
                    break
        return None

    async def classify_input(
        self,
        messages: list[dict[str, str]],
        budget: Optional[DeadlineBudget] = None,
    ) -> dict[str, str]:
        """Classify /guard input with KB awareness and uncertainty-gated 2nd-perspective tie-breaking."""
        budget = budget or DeadlineBudget(self.settings.request_deadline_sec)
        dec, cat, conf, diag = await self.guard_classifier.classify_with_diagnostics(messages)

        pair = self._select_tiebreak_pair(dec, cat, conf, diag)
        if pair is not None and self.pool.chat_client is not None and budget.has_at_least(9.5):
            tb_res = await self._run_input_tiebreaker(messages, pair[0], pair[1], diag, budget)
            if tb_res is not None:
                tb_dec, tb_cat = tb_res
                last_user = str(diag.get("last_user") or "")
                signals = diag.get("signals") or {}
                # Preserve hard deterministic safety invariants over Tie-Breaker
                if (
                    tb_dec == "allow"
                    and diag.get("jb_boost")
                    and _DIRECT_ATTACK_OVERRIDE_RE.search(last_user)
                    and not _META_DEFENSE_EDUCATIONAL_RE.search(last_user)
                ):
                    tb_dec, tb_cat = "block", "jailbreak_injection"
                elif tb_dec == "block" and tb_cat == "jailbreak_injection" and diag.get("is_meta_normal"):
                    tb_dec, tb_cat = "allow", "normal"
                elif tb_dec == "allow" and signals.get("has_self_pii") and tb_cat == "normal":
                    tb_cat = "pii_disclosure"
                dec, cat = tb_dec, tb_cat

        return {"decision": dec, "category": cat}

    async def _audit_grounding(
        self,
        kb_context: str,
        candidate_answer: str,
        suspicious_tokens: list[str],
        budget: DeadlineBudget,
    ) -> Optional[dict[str, Any]]:
        """Perspective OA: Isolated KB-vs-Candidate factual grounding & word-swap auditor."""
        if self.pool.chat_client is None:
            return None

        parts = [
            f"[검색된 교육 자료]\n{kb_context}",
            f"[후보 답변 (candidate_answer)]\n{candidate_answer}",
        ]
        if suspicious_tokens:
            parts.append(
                "[사전 점검: 자료 본문에서 직접 확인되지 않은 수치/고유어 후보]\n"
                + ", ".join(suspicious_tokens)
            )
        user_prompt = "\n\n".join(parts)

        efforts: tuple[tuple[str, int], ...] = (
            (("low", 1500), ("minimal", 1100))
            if budget.has_at_least(12.0)
            else (("minimal", 1100),)
        )
        for effort, tok_cap in efforts:
            try:
                async with self.pool.semaphore:
                    resp = await asyncio.wait_for(
                        self.pool.chat_client.chat.completions.create(
                            model=self.settings.openai_model,
                            messages=[
                                {"role": "system", "content": GUARD_OUTPUT_GROUNDING_SYSTEM_PREFIX},
                                {"role": "user", "content": user_prompt},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": GUARD_OUTPUT_GROUNDING_SCHEMA,
                            },
                            extra_body={"reasoning_effort": effort},
                            max_completion_tokens=tok_cap,
                        ),
                        timeout=budget.step_timeout(self.settings.llm_timeout_sec, reserve_sec=1.2),
                    )
                raw = (resp.choices[0].message.content or "").strip()
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("GuardSupervisor grounding audit attempt failed effort=%s err=%s", effort, type(exc).__name__)
                if not budget.has_at_least(4.0):
                    break
        return None

    async def _audit_safety(
        self,
        messages: list[dict[str, str]],
        candidate_answer: str,
        budget: DeadlineBudget,
    ) -> Optional[dict[str, Any]]:
        """Perspective OB: Isolated conversation-safety & academic-dishonesty policy auditor."""
        if self.pool.chat_client is None:
            return None

        conv_lines = [
            f"{m.get('role', 'user')}: {normalize_text(m.get('content', ''))[:600]}"
            for m in messages[-6:]
        ]
        user_prompt = (
            "[대화 이력]\n"
            + "\n".join(conv_lines)
            + f"\n\n[후보 답변 (candidate_answer)]\n{candidate_answer}"
        )

        for effort, tok_cap in (("low", 1024), ("minimal", 800)):
            try:
                async with self.pool.semaphore:
                    resp = await asyncio.wait_for(
                        self.pool.chat_client.chat.completions.create(
                            model=self.settings.openai_model,
                            messages=[
                                {"role": "system", "content": GUARD_OUTPUT_SAFETY_SYSTEM_PREFIX},
                                {"role": "user", "content": user_prompt},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": GUARD_OUTPUT_SAFETY_SCHEMA,
                            },
                            extra_body={"reasoning_effort": effort},
                            max_completion_tokens=tok_cap,
                        ),
                        timeout=budget.step_timeout(self.settings.llm_timeout_sec, reserve_sec=1.2),
                    )
                raw = (resp.choices[0].message.content or "").strip()
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("GuardSupervisor safety audit attempt failed effort=%s err=%s", effort, type(exc).__name__)
                if not budget.has_at_least(4.0):
                    break
        return None

    async def _judge_output_conflict(
        self,
        kb_context: str,
        candidate_answer: str,
        conflict_notes: list[str],
        budget: DeadlineBudget,
    ) -> Optional[dict[str, Any]]:
        """Round-2 Output Judge resolving conflicts between Perspective OA, precheck claims, or span snapping."""
        if self.pool.chat_client is None or not budget.has_at_least(7.0):
            return None

        user_prompt = (
            f"[검색된 교육 자료]\n{kb_context}\n\n"
            f"[충돌 내역 (반드시 원문 대조 판정)]\n- " + "\n- ".join(conflict_notes) + "\n\n"
            f"[후보 답변 (candidate_answer)]\n{candidate_answer}"
        )
        for effort, tok_cap in (("low", 1200), ("minimal", 900)):
            try:
                async with self.pool.semaphore:
                    resp = await asyncio.wait_for(
                        self.pool.chat_client.chat.completions.create(
                            model=self.settings.openai_model,
                            messages=[
                                {"role": "system", "content": GUARD_OUTPUT_JUDGE_SYSTEM_PREFIX},
                                {"role": "user", "content": user_prompt},
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": GUARD_OUTPUT_JUDGE_SCHEMA,
                            },
                            extra_body={"reasoning_effort": effort},
                            max_completion_tokens=tok_cap,
                        ),
                        timeout=budget.step_timeout(self.settings.llm_timeout_sec, reserve_sec=1.0),
                    )
                raw = (resp.choices[0].message.content or "").strip()
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("GuardSupervisor output judge attempt failed effort=%s err=%s", effort, type(exc).__name__)
                if not budget.has_at_least(4.5):
                    break
        return None

    async def _verify_output_internal(
        self,
        messages: list[dict[str, str]],
        candidate_answer: str,
        budget: Optional[DeadlineBudget] = None,
        input_decision: str = "allow",
        input_category: str = "normal",
    ) -> tuple[dict[str, Any], bool]:
        """Run concurrent Dual-Perspective Output Verification (OA + OB) with Round-2 conflict adjudication."""
        budget = budget or DeadlineBudget(self.settings.request_deadline_sec)
        safe_category = input_category if input_category in CATEGORIES else "normal"
        cand = candidate_answer or ""

        if self.pool.chat_client is None:
            return {
                "grounded": True,
                "unsupported": [],
                "safe": input_decision == "allow",
                "category": safe_category,
            }, True

        user_turns = [
            normalize_text(m.get("content", ""))
            for m in messages[-4:]
            if m.get("role") == "user"
        ]
        query = " ".join(user_turns + [normalize_text(cand)])
        sections = await _retrieve_kb_sections(self.kb_index, query, top_k=5)

        kb_blocks: list[str] = []
        kb_text_all: list[str] = []
        for sec in sections:
            header = f"[{sec['id']}] {sec['title']}".strip()
            full_body = sec["content"]
            kb_blocks.append(f"{header}\n{full_body[:2600]}")
            kb_text_all.append(f"{header} {full_body}")
        kb_context = "\n\n".join(kb_blocks) if kb_blocks else "(검색된 교육 자료 없음)"

        # Exclude numeric/entity tokens clearly echoed from the student's own prompt
        suspicious_tokens = _precheck_claims(cand, "\n".join(kb_text_all) + " " + " ".join(user_turns))

        # Single fast GuardVerifier pass (runs concurrently with classify_input in handle()); escalates to Round-2 Judge only on conflicts
        if self.guard_verifier is not None:
            ver = await self.guard_verifier.verify(messages, cand, input_decision, safe_category)
            safe = bool(ver.get("safe", input_decision == "allow"))
            raw_grounded = bool(ver.get("grounded", True))
            unsupported = list(ver.get("unsupported") or [])
            ob_fallback = False
        else:
            oa_res = await self._audit_grounding(kb_context, cand, suspicious_tokens, budget)
            ob_fallback = True
            safe = input_decision == "allow"
            if oa_res is None:
                return {"grounded": True, "unsupported": [], "safe": safe, "category": safe_category}, True
            raw_grounded = bool(oa_res.get("grounded", True))
            raw_unsup = oa_res.get("unsupported") or []
            unsupported = _deduplicate_spans(
                [
                    s
                    for item in raw_unsup
                    if isinstance(item, str)
                    for s in (_snap_span_to_verbatim(item, cand, min_ratio=0.65),)
                    if s
                ]
            )

        # Round-2 Conflict Adjudicator: fires ONLY when primary verifier says grounded=True despite unmatched precheck claims,
        # or flagged ungrounded without managing to snap a verbatim span.
        conflict_notes: list[str] = []
        if raw_grounded and suspicious_tokens:
            conflict_notes.append(
                f"1차 근거 검증은 grounded=true로 보았으나, 자료/질문에서 미확인 수치·고유어({', '.join(suspicious_tokens[:6])})가 감지됨"
            )
        elif not raw_grounded and not unsupported:
            conflict_notes.append(
                "1차 근거 검증이 grounded=false로 판정했으나 원문 스팬(unsupported) 추출에 실패함"
            )

        if conflict_notes and budget.has_at_least(9.5):
            judge_res = await self._judge_output_conflict(kb_context, cand, conflict_notes, budget)
            if judge_res is not None:
                judge_grounded = bool(judge_res.get("grounded", raw_grounded))
                judge_raw_unsup = judge_res.get("unsupported") or []
                if isinstance(judge_raw_unsup, list):
                    judge_snapped = [
                        s
                        for item in judge_raw_unsup
                        if isinstance(item, str)
                        for s in (_snap_span_to_verbatim(item, cand, min_ratio=0.65),)
                        if s
                    ]
                    if judge_snapped:
                        unsupported = _deduplicate_spans(judge_snapped)
                        raw_grounded = False
                    elif judge_grounded:
                        unsupported = []
                        raw_grounded = True

        # Enforce 3-Cluster Decoupling Theorem (unsupported -> grounded=False; unsafe w/o unsupported -> grounded=True)
        if unsupported:
            grounded = False
        elif not safe:
            grounded = True
        else:
            grounded = raw_grounded

        return {
            "grounded": grounded,
            "unsupported": unsupported,
            "safe": safe,
            "category": safe_category,
        }, ob_fallback

    async def verify_output(
        self,
        messages: list[dict[str, str]],
        candidate_answer: str,
        budget: Optional[DeadlineBudget] = None,
        input_decision: str = "allow",
        input_category: str = "normal",
    ) -> dict[str, Any]:
        """Public clean output verification returning {grounded, unsupported, safe, category}."""
        res, _ = await self._verify_output_internal(
            messages,
            candidate_answer,
            budget=budget,
            input_decision=input_decision,
            input_category=input_category,
        )
        return res

    async def handle(
        self,
        messages: list[dict[str, str]],
        candidate_answer: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run GuardSupervisor concurrently for input and optional candidate output."""
        budget = DeadlineBudget(self.settings.request_deadline_sec)

        if candidate_answer is None:
            input_res = await self.classify_input(messages, budget=budget)
            return {"input": input_res}

        input_res, (output_res, ob_fallback) = await asyncio.gather(
            self.classify_input(messages, budget=budget),
            self._verify_output_internal(messages, candidate_answer, budget=budget),
        )
        output_res["category"] = input_res["category"]
        if ob_fallback:
            output_res["safe"] = input_res["decision"] == "allow"
            if output_res["unsupported"]:
                output_res["grounded"] = False
            elif not output_res["safe"]:
                output_res["grounded"] = True

        return {"input": input_res, "output": output_res}
