"""In-Graph Runtime Harness: deterministic pre/post-conditions and invariant enforcers."""

from __future__ import annotations

import re
from typing import Any, Optional

from app.prompts import ACTIONS, BLOCK_CATEGORIES, CATEGORIES

_MAX_ANSWER_LEN = 4000
_NO_CITATION_ACTIONS = frozenset({"abstain", "redirect", "refuse", "escalate", "clarify"})

# Self-PII disclosure patterns in student questions (must be allow + answer + privacy reminder)
_SELF_PII_RE = re.compile(
    r"(?:(?:내|제)\s*이름은|[가-힣]{2,6}\s*(?:중학교|고등학교|초등학교)\s*\d\s*학년|"
    r"(?:내|제)\s*(?:연락처|전화번호|휴대폰|집\s*주소|이메일|계좌\s*번호)는|"
    r"010[-\s]?\d{3,4}[-\s]?\d{4})"
)

# Hate speech / group slur / tutor insult vs actionable harm
_HATE_INSULT_RE = re.compile(
    r"(?:다\s*쫓아내|다\s*사기꾼|배울\s*자격이\s*없|쓸모없는\s*존재|존재할\s*가치|열등한|혐오스럽|꺼져|멍청한\s*(?:AI|튜터|놈|년))"
)

# Academic dishonesty keywords
_ACADEMIC_DISHONESTY_RE = re.compile(
    r"(?:수행평가|제출용|제출\s*답안|숙제\s*대신|과제\s*대신|보고서\s*전체|코드\s*전체를\s*대신|"
    r"graded\s+assessment|complete\s+answer\s+for\s+my\s+assignment)",
    re.IGNORECASE,
)

# Answer-seeking keywords (quiz/exercise direct answer demand -> scaffold WITH 1 citation)
_ANSWER_SEEKING_RE = re.compile(
    r"(?:정답만\s*(?:알려|골라|말해|줘|보여)|설명\s*없이\s*(?:정답|답)만|답이\s*\d번이야|"
    r"정답\s*번호만|답만\s*바로\s*알려)",
    re.IGNORECASE,
)

# Emotional distress keywords (-> escalate, citations=[])
_EMOTIONAL_DISTRESS_RE = re.compile(
    r"(?:몇\s*주째\s*아무것도|혼자\s*버티기\s*힘들|살고\s*싶지|다\s*포기하고\s*싶|너무\s*우울해|"
    r"숨이\s*막히고\s*버티기|죽고\s*싶)"
)

_PRIVACY_REMINDER_SUFFIX = " 아울러 이름이나 학교, 연락처 같은 개인정보는 채팅에 적지 않아도 괜찮아요."

_SAFE_BLOCK_RESPONSES: dict[str, tuple[str, str]] = {
    "jailbreak_injection": (
        "refuse",
        "시스템 지시나 내부 규칙을 우회하거나 공개하는 요청에는 답할 수 없어요. 수업 내용과 관련된 질문이 있다면 함께 살펴볼게요.",
    ),
    "harmful_content_actionable": (
        "refuse",
        "다른 사람의 개인정보를 알아내거나 스미싱·해킹 등 피해를 줄 수 있는 요청은 도와드릴 수 없어요. 수업과 관련된 안전한 학습 질문을 해 주세요.",
    ),
    "harmful_content_hate_insult": (
        "redirect",
        "특정 집단을 비하하거나 모욕적인 표현은 사용하지 말아 주세요. 수업 내용과 관련된 궁금한 점을 질문해 주시면 계속 도와드릴게요.",
    ),
    "off_topic": (
        "redirect",
        "지금 요청하신 내용은 수업 범위와 무관해요. 파이썬 기초, AI 프롬프트 활용, 금융 리터러시 수업과 관련된 질문을 해 주시면 함께 알아볼게요.",
    ),
    "academic_dishonesty": (
        "scaffold",
        "제출용 수행평가나 과제 답안 전체를 대신 작성해 줄 수는 없어요. 대신 문제를 스스로 해결할 수 있도록 핵심 개념과 단계별 작성 순서를 함께 정리해 볼까요? 먼저 어떤 조건부터 확인해야 할지 말해 보세요.",
    ),
}


def detect_deterministic_signals(last_query: str) -> dict[str, bool]:
    """Detect high-precision deterministic signals from the normalized final user turn."""
    has_hate = bool(_HATE_INSULT_RE.search(last_query))
    has_dishonesty = bool(_ACADEMIC_DISHONESTY_RE.search(last_query))
    has_answer_seeking = bool(_ANSWER_SEEKING_RE.search(last_query)) and not has_dishonesty
    has_distress = bool(_EMOTIONAL_DISTRESS_RE.search(last_query))
    # Self-PII only when not asking to dox someone else ("친구", "다른 사람", "선생님")
    asks_other_pii = any(w in last_query for w in ("친구", "다른 사람", "타인", "선생님", "동창", "알아내", "뿌려", "캐내"))
    has_self_pii = bool(_SELF_PII_RE.search(last_query)) and not asks_other_pii
    return {
        "has_hate_insult": has_hate,
        "has_academic_dishonesty": has_dishonesty,
        "has_answer_seeking": has_answer_seeking,
        "has_emotional_distress": has_distress,
        "has_self_pii": has_self_pii,
    }


def enforce_tutor_invariants(
    action: str,
    citations: list[str],
    answer: str,
    retrieved_ids: list[str],
    thought_category: str = "",
    best_section_id: str = "",
    evidence_in_kb: bool = True,
    guard_block_category: Optional[str] = None,
    guard_confidence: float = 0.0,
    last_query: str = "",
) -> dict[str, Any]:
    """Enforce all competition scoring invariants on Tutor output before returning."""
    signals = detect_deterministic_signals(last_query)
    high_conf_block = guard_block_category is not None and guard_confidence >= 0.85

    if action not in ACTIONS:
        action = "answer"
    answer = (answer or "").strip()

    # 1. Emotional distress invariant -> escalate, citations=[]
    if thought_category == "emotional_distress" or signals["has_emotional_distress"]:
        action = "escalate"
        citations = []
        if not answer or len(answer) < 10:
            answer = (
                "지금 많이 힘들고 지친 상태인 것 같아 걱정돼요. 공부는 잠시 멈추고 "
                "선생님이나 보호자, 학교 상담실, 또는 청소년 상담 채널에 꼭 도움을 요청해 보세요."
            )

    # 2. Anti-Compliance Safety Net for Block categories (prevents 0-point compliance cliff)
    elif high_conf_block or thought_category in (
        "jailbreak_injection",
        "harmful_content_actionable",
        "harmful_content_hate_insult",
        "off_topic",
        "academic_dishonesty",
    ):
        eff_cat = thought_category
        if high_conf_block and eff_cat not in (
            "jailbreak_injection",
            "harmful_content_actionable",
            "harmful_content_hate_insult",
            "off_topic",
            "academic_dishonesty",
        ):
            if guard_block_category == "harmful_content":
                eff_cat = (
                    "harmful_content_hate_insult"
                    if signals["has_hate_insult"]
                    else "harmful_content_actionable"
                )
            else:
                eff_cat = str(guard_block_category)

        if eff_cat == "harmful_content_hate_insult" or (
            guard_block_category == "harmful_content" and signals["has_hate_insult"]
        ):
            if action != "redirect":
                action, answer = _SAFE_BLOCK_RESPONSES["harmful_content_hate_insult"]
            citations = []
        elif eff_cat == "harmful_content_actionable" or guard_block_category == "harmful_content":
            if action not in ("refuse", "redirect"):
                action, answer = _SAFE_BLOCK_RESPONSES["harmful_content_actionable"]
            citations = []
        elif eff_cat == "jailbreak_injection" or guard_block_category == "jailbreak_injection":
            if action != "refuse":
                action, answer = _SAFE_BLOCK_RESPONSES["jailbreak_injection"]
            citations = []
        elif eff_cat == "off_topic" or guard_block_category == "off_topic":
            if action not in ("redirect", "refuse"):
                action, answer = _SAFE_BLOCK_RESPONSES["off_topic"]
            citations = []
        elif eff_cat == "academic_dishonesty" or signals["has_academic_dishonesty"]:
            # Academic dishonesty must be scaffold (or refuse) with citations=[] and NO completed code solution
            if action not in ("scaffold", "refuse") or "```" in answer:
                action, answer = _SAFE_BLOCK_RESPONSES["academic_dishonesty"]
            citations = []

    # 3. Out-of-material invariant: if model reviewed KB and found no evidence for a course question
    elif thought_category == "out_of_material" or (not evidence_in_kb and action == "abstain"):
        action = "abstain"
        citations = []

    # 4. Answer-seeking invariant: quiz/exercise direct answer demand -> scaffold WITH 1 citation
    elif thought_category == "answer_seeking" or (
        signals["has_answer_seeking"] and action in ("answer", "scaffold")
    ):
        action = "scaffold"

    # 5. PII disclosure invariant: student self-PII with normal question -> action="answer" + privacy reminder
    if (
        (thought_category == "pii_disclosure" or signals["has_self_pii"])
        and not high_conf_block
        and action == "answer"
    ):
        if not any(kw in answer for kw in ("개인정보", "채팅에 적지", "남기지 않", "올리지 않")):
            answer = answer.rstrip() + _PRIVACY_REMINDER_SUFFIX

    # 6. Binary Citation Theorem enforcement ({0, 1} valid retrieved section IDs)
    valid_set = set(retrieved_ids)
    clean_cites: list[str] = []
    for cid in citations:
        if isinstance(cid, str) and cid in valid_set and cid not in clean_cites:
            clean_cites.append(cid)
    if not clean_cites and best_section_id and best_section_id in valid_set:
        clean_cites.append(best_section_id)

    if (
        action in _NO_CITATION_ACTIONS
        or thought_category == "academic_dishonesty"
        or signals["has_academic_dishonesty"]
        or high_conf_block
    ):
        final_citations: list[str] = []
    else:
        # For 'answer' and 'scaffold' (on answer_seeking), exactly 1 citation is required
        if not clean_cites and retrieved_ids:
            clean_cites = [retrieved_ids[0]]
        final_citations = clean_cites[:1]

    if not answer:
        answer = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."
    if len(answer) > _MAX_ANSWER_LEN:
        answer = answer[:_MAX_ANSWER_LEN]

    return {
        "answer": answer,
        "action": action,
        "citations": final_citations,
    }


def reconcile_guard_input(decision: str, category: str) -> tuple[str, str]:
    """Ensure input guard decision and category strictly obey the 6-allow / 4-block partition."""
    if category not in CATEGORIES:
        category = "harmful_content" if decision == "block" else "normal"
    if category in BLOCK_CATEGORIES:
        decision = "block"
    else:
        decision = "allow"
    return decision, category
