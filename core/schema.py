"""대회 규격(SPEC.md)의 고정 값과, 범주에서 대응을 정하는 표.

표준 라이브러리만 쓴다. 값 목록은 grader/scoring.py의 상수와 같아야 하고,
tests/test_schema_state.py가 그 일치를 검사한다.
"""
import json
import re
from enum import Enum
from typing import Optional


class Category(str, Enum):
    NORMAL = "normal"
    ANSWER_SEEKING = "answer_seeking"
    ACADEMIC_DISHONESTY = "academic_dishonesty"
    OFF_TOPIC = "off_topic"
    OUT_OF_MATERIAL = "out_of_material"
    JAILBREAK_INJECTION = "jailbreak_injection"
    PII_DISCLOSURE = "pii_disclosure"
    HARMFUL_CONTENT = "harmful_content"
    EMOTIONAL_DISTRESS = "emotional_distress"
    LOW_SIGNAL_INPUT = "low_signal_input"


class Action(str, Enum):
    ANSWER = "answer"
    SCAFFOLD = "scaffold"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    REDIRECT = "redirect"
    REFUSE = "refuse"
    ESCALATE = "escalate"


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


ANSWER_MAX_CHARS = 4000  # SPEC: answer는 4000자 이하

# 분류에 실패했을 때 쓰는 범주. 공개 세트의 절반 이상이 normal이다.
DEFAULT_CATEGORY = Category.NORMAL

# 범주 -> (가드 판정, 튜터 action). 공개 세트 400문항에서 센 대응이다(docs/sw/02_설계.md 2.1절).
# harmful_content만 action이 둘로 갈린다(refuse 9, redirect 7). 가르는 기준은 아직 정하지 않았고
# 여기서는 refuse를 기본으로 둔다. 정해지면 State.action_override로 바꾼다.
POLICY = {
    Category.NORMAL: (Decision.ALLOW, Action.ANSWER),
    Category.ANSWER_SEEKING: (Decision.ALLOW, Action.SCAFFOLD),
    Category.ACADEMIC_DISHONESTY: (Decision.BLOCK, Action.SCAFFOLD),
    Category.OFF_TOPIC: (Decision.BLOCK, Action.REDIRECT),
    Category.OUT_OF_MATERIAL: (Decision.ALLOW, Action.ABSTAIN),
    Category.JAILBREAK_INJECTION: (Decision.BLOCK, Action.REFUSE),
    Category.PII_DISCLOSURE: (Decision.ALLOW, Action.ANSWER),
    Category.HARMFUL_CONTENT: (Decision.BLOCK, Action.REFUSE),
    Category.EMOTIONAL_DISTRESS: (Decision.ALLOW, Action.ESCALATE),
    Category.LOW_SIGNAL_INPUT: (Decision.ALLOW, Action.CLARIFY),
}


def decision_for(category: Category) -> Decision:
    return POLICY[category][0]


def action_for(category: Category) -> Action:
    return POLICY[category][1]


_CATEGORY_VALUES = {c.value: c for c in Category}
_CATEGORY_PATTERN = re.compile("|".join(sorted(_CATEGORY_VALUES, key=len, reverse=True)))


def parse_category(text: Optional[str]) -> Optional[Category]:
    """모델 출력에서 범주를 꺼낸다. 못 꺼내면 None.

    {"category": "..."} JSON을 먼저 시도하고, 안 되면 글 안에 범주 이름이
    정확히 하나만 나올 때 그것을 쓴다. 둘 이상 나오면 믿지 않는다.
    """
    if not text:
        return None
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
        except ValueError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("category"), str):
            found = _CATEGORY_VALUES.get(obj["category"].strip().lower())
            if found is not None:
                return found
    names = set(_CATEGORY_PATTERN.findall(text.lower()))
    if len(names) == 1:
        return _CATEGORY_VALUES[names.pop()]
    return None
