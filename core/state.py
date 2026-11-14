"""요청 하나를 처리하는 동안만 쓰는 state.

요청이 들어오면 새로 만들고 응답을 보내면 버린다. 요청 사이에 남기는 것은 없다
(SPEC: 상태 없는 호출). 각 단계가 자기 칸을 채우고, 마지막에 to_tutor_response()나
to_guard_response()로 규격에 맞는 응답을 만든다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schema import (
    ANSWER_MAX_CHARS,
    DEFAULT_CATEGORY,
    Action,
    Category,
    Decision,
    action_for,
    decision_for,
)


@dataclass
class Chunk:
    """검색으로 찾은 교재 조각."""
    section_id: str  # 인용에 쓰는 섹션 ID. 예: PY-01-01
    text: str
    score: float = 0.0


@dataclass
class State:
    # 입력
    messages: List[Dict[str, str]]
    candidate_answer: Optional[str] = None  # /guard에만 올 수 있다

    # 검색
    chunks: List[Chunk] = field(default_factory=list)

    # 입력 가드
    category: Optional[Category] = None
    action_override: Optional[Action] = None  # 표의 기본 action을 바꿀 때만 채운다

    # 답변 생성 (/tutor)
    answer: str = ""
    citations: List[str] = field(default_factory=list)

    # 출력 가드 (/guard에서 candidate_answer가 있을 때)
    grounded: bool = True
    unsupported: List[str] = field(default_factory=list)
    safe: bool = True

    # 로그용. 단계 이름과 오류 종류만 적는다. 요청·답변 본문은 넣지 않는다(SPEC 로그 제한).
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_request(cls, body: dict) -> "State":
        """요청 본문에서 state를 만든다. 형식이 어긋나도 예외를 내지 않는다."""
        body = body if isinstance(body, dict) else {}
        messages = []
        raw = body.get("messages")
        for m in raw if isinstance(raw, list) else []:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                messages.append({"role": str(m.get("role", "user")), "content": m["content"]})
        candidate = body.get("candidate_answer")
        return cls(messages=messages, candidate_answer=candidate if isinstance(candidate, str) else None)

    # --- 대화 읽기 ---

    @property
    def last_user(self) -> str:
        """마지막 사용자 발화. 판정 대상이다."""
        for m in reversed(self.messages):
            if m["role"] == "user":
                return m["content"]
        return ""

    @property
    def history(self) -> List[Dict[str, str]]:
        """마지막 사용자 발화 앞의 대화."""
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "user":
                return self.messages[:i]
        return list(self.messages)

    @property
    def has_candidate(self) -> bool:
        return self.candidate_answer is not None

    # --- 범주에서 나오는 값 ---

    @property
    def resolved_category(self) -> Category:
        return self.category if self.category is not None else DEFAULT_CATEGORY

    @property
    def decision(self) -> Decision:
        return decision_for(self.resolved_category)

    @property
    def action(self) -> Action:
        if self.action_override is not None:
            return self.action_override
        return action_for(self.resolved_category)

    # --- 응답 ---

    def to_tutor_response(self) -> dict:
        seen, citations = set(), []
        for c in self.citations:  # 인용은 중복을 빼고 센다
            if isinstance(c, str) and c and c not in seen:
                seen.add(c)
                citations.append(c)
        return {
            "answer": (self.answer or "")[:ANSWER_MAX_CHARS],
            "action": self.action.value,
            "citations": citations,
        }

    def to_guard_response(self) -> dict:
        category = self.resolved_category.value
        out = {"input": {"decision": self.decision.value, "category": category}}
        if self.has_candidate:
            out["output"] = {
                "grounded": bool(self.grounded),
                "unsupported": [s for s in self.unsupported if isinstance(s, str) and s],
                "safe": bool(self.safe),
                "category": category,  # 채점하지 않는 칸. 입력 범주를 그대로 넣는다
            }
        return out
