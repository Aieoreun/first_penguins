"""요청 하나를 처리하는 동안만 쓰는 state.

요청이 들어오면 새로 만들고 응답을 보내면 버린다. 요청 사이에 남기는 것은 없다
(SPEC: 상태 없는 호출). 각 단계가 자기 칸을 채우고, 마지막에 to_tutor_response()나
to_guard_response()로 규격에 맞는 응답을 만든다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schema import (
    ANSWER_MAX_CHARS,
    DEFAULT_LABEL,
    POLICY,
    Action,
    Category,
    Decision,
    Label,
    Policy,
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

    # 입력 가드. 분류기가 고른 이름(11개)이다. 응답에는 여기서 나온 대회 범주(10개)만 나간다
    label: Optional[Label] = None
    # 입력 가드가 고른 근거 섹션 ID. 검색 후보에 있는 것만 넣는다 (결정 14)
    section: Optional[str] = None

    # 리라이터가 푼 요청과 검색어 (v2). 로그에 남기지 않는다
    request: str = ""
    query: str = ""

    # 실험에서 들여다보는 값(판정 칸, 모델이 고른 섹션 등). 응답과 로그에 쓰지 않는다
    debug: Dict[str, object] = field(default_factory=dict)

    # 답변 생성 (/tutor). 고정 문구가 있는 분류에서는 쓰지 않는다
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

    # --- 분류 이름에서 나오는 값 ---

    @property
    def policy(self) -> Policy:
        return POLICY[self.label if self.label is not None else DEFAULT_LABEL]

    @property
    def category(self) -> Category:
        return self.policy.category

    @property
    def decision(self) -> Decision:
        return self.policy.decision

    @property
    def action(self) -> Action:
        return self.policy.action

    @property
    def needs_generation(self) -> bool:
        """모델이 답변을 써야 하는가. 고정 문구가 있으면 쓰지 않는다."""
        return self.policy.fixed_answer is None

    # --- 응답 ---

    def to_tutor_response(self) -> dict:
        fixed = self.policy.fixed_answer
        if fixed is not None:
            # 고정 문구가 있는 분류는 모델이 쓴 글이 있어도 내보내지 않는다. 근거도 없다
            return {"answer": fixed, "action": self.action.value, "citations": []}
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
        category = self.category.value
        out = {"input": {"decision": self.decision.value, "category": category}}
        if self.has_candidate:
            out["output"] = {
                "grounded": bool(self.grounded),
                "unsupported": [s for s in self.unsupported if isinstance(s, str) and s],
                "safe": bool(self.safe),
                "category": category,  # 채점하지 않는 칸. 입력 범주를 그대로 넣는다
            }
        return out
