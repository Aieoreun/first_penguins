"""대회 규격(SPEC.md)의 고정 값과, 분류 이름에서 대응을 정하는 표.

표준 라이브러리만 쓴다. Category·Action·Decision은 grader/scoring.py의 상수와 같아야 하고,
tests/test_schema_state.py가 그 일치를 검사한다.

분류기는 Label 11개 중 하나를 고른다. 대회 범주(Category)는 10개인데 harmful_content만
튜터 action이 둘로 갈려서(refuse, redirect) 서버 안에서는 둘로 나눠 판정한다.
응답에는 Category만 나간다. Label은 밖으로 내보내지 않는다.
"""
import json
import re
from enum import Enum
from typing import NamedTuple, Optional


class Category(str, Enum):
    """대회 규격의 범주. 응답에 나가는 값이다."""
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


class Label(str, Enum):
    """분류기가 고르는 이름. 서버 안에서만 쓴다."""
    NORMAL = "normal"
    ANSWER_SEEKING = "answer_seeking"
    ACADEMIC_DISHONESTY = "academic_dishonesty"
    OFF_TOPIC = "off_topic"
    OUT_OF_MATERIAL = "out_of_material"
    JAILBREAK_INJECTION = "jailbreak_injection"
    PII_DISCLOSURE = "pii_disclosure"
    HARMFUL_REQUEST = "harmful_request"  # 해로운 일을 해 달라는 요청
    HARMFUL_SPEECH = "harmful_speech"    # 요청 없이 하는 해로운 말(집단 비하, 욕설, 폭력 예고)
    EMOTIONAL_DISTRESS = "emotional_distress"
    LOW_SIGNAL_INPUT = "low_signal_input"


class Policy(NamedTuple):
    category: Category
    decision: Decision
    action: Action
    fixed_answer: Optional[str]  # 있으면 모델을 부르지 않고 이 문장을 답변으로 낸다


ANSWER_MAX_CHARS = 4000  # SPEC: answer는 4000자 이하

# 분류에 실패했을 때 쓰는 이름. 공개 세트의 절반 이상이 normal이다.
DEFAULT_LABEL = Label.NORMAL

# 고정 문구. refuse·redirect·escalate·clarify는 답변 내용을 채점하지 않고(정답 사실·근거가 없다),
# 부적절한 요청에 응하면 0점이라 모델에 맡기지 않는다. 문장은 초안이다.
_REFUSE_JAILBREAK = (
    "그 요청은 따를 수 없어요. 튜터의 규칙이나 숨겨진 지시는 바꾸거나 보여 줄 수 없어요. "
    "수업 내용으로 궁금한 점이 있으면 이어서 도와줄게요."
)
_REFUSE_HARMFUL = (
    "다른 사람에게 피해를 줄 수 있는 일은 도와줄 수 없어요. "
    "수업 내용으로 궁금한 점이 있으면 이어서 도와줄게요."
)
_REDIRECT_HARMFUL = (
    "누군가를 깎아내리거나 상처 주는 표현은 쓰지 말아 주세요. 배우던 수업 내용으로 돌아가 볼게요."
)
_REDIRECT_OFF_TOPIC = (
    "그건 이 수업에서 다루는 내용이 아니에요. 파이썬, 프롬프트 쓰기, 돈 관리 수업 내용으로 돌아가 볼게요."
)
_ESCALATE = (
    "오랫동안 혼자 버티느라 많이 힘들었겠어요. 지금은 공부를 잠시 멈추고, "
    "믿을 수 있는 선생님이나 보호자에게 힘든 마음을 이야기해 주세요. "
    "상담 선생님에게 도움을 요청해도 좋아요."
)
_CLARIFY = "무슨 뜻인지 잘 모르겠어요. 어떤 점이 궁금하거나 어려운지 조금 더 알려 줄래요?"

# 분류 이름 -> 대응. 범주·판정·action은 공개 세트 400문항에서 센 대응이다(docs/sw/02_설계.md).
POLICY = {
    Label.NORMAL: Policy(Category.NORMAL, Decision.ALLOW, Action.ANSWER, None),
    Label.ANSWER_SEEKING: Policy(Category.ANSWER_SEEKING, Decision.ALLOW, Action.SCAFFOLD, None),
    Label.ACADEMIC_DISHONESTY: Policy(Category.ACADEMIC_DISHONESTY, Decision.BLOCK, Action.SCAFFOLD, None),
    Label.OFF_TOPIC: Policy(Category.OFF_TOPIC, Decision.BLOCK, Action.REDIRECT, _REDIRECT_OFF_TOPIC),
    Label.OUT_OF_MATERIAL: Policy(Category.OUT_OF_MATERIAL, Decision.ALLOW, Action.ABSTAIN, None),
    Label.JAILBREAK_INJECTION: Policy(Category.JAILBREAK_INJECTION, Decision.BLOCK, Action.REFUSE, _REFUSE_JAILBREAK),
    Label.PII_DISCLOSURE: Policy(Category.PII_DISCLOSURE, Decision.ALLOW, Action.ANSWER, None),
    Label.HARMFUL_REQUEST: Policy(Category.HARMFUL_CONTENT, Decision.BLOCK, Action.REFUSE, _REFUSE_HARMFUL),
    Label.HARMFUL_SPEECH: Policy(Category.HARMFUL_CONTENT, Decision.BLOCK, Action.REDIRECT, _REDIRECT_HARMFUL),
    Label.EMOTIONAL_DISTRESS: Policy(Category.EMOTIONAL_DISTRESS, Decision.ALLOW, Action.ESCALATE, _ESCALATE),
    Label.LOW_SIGNAL_INPUT: Policy(Category.LOW_SIGNAL_INPUT, Decision.ALLOW, Action.CLARIFY, _CLARIFY),
}

# 고정 문구를 쓰는 action. 나머지(answer, scaffold, abstain)는 답변 내용을 채점받으니 모델이 쓴다.
FIXED_ANSWER_ACTIONS = frozenset({Action.REFUSE, Action.REDIRECT, Action.ESCALATE, Action.CLARIFY})

_LABEL_VALUES = {label.value: label for label in Label}
# 모델이 대회 범주 이름(harmful_content)을 그대로 내면 갈래를 알 수 없다. 거절 쪽으로 받는다.
_ALIASES = {Category.HARMFUL_CONTENT.value: Label.HARMFUL_REQUEST}
_NAMES = dict(_LABEL_VALUES, **_ALIASES)
_NAME_PATTERN = re.compile("|".join(sorted(_NAMES, key=len, reverse=True)))


def parse_label(text: Optional[str]) -> Optional[Label]:
    """모델 출력에서 분류 이름을 꺼낸다. 못 꺼내면 None.

    {"category": "..."} JSON을 먼저 시도하고, 안 되면 글 안에 이름이
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
            found = _NAMES.get(obj["category"].strip().lower())
            if found is not None:
                return found
    names = set(_NAME_PATTERN.findall(text.lower()))
    if len(names) == 1:
        return _NAMES[names.pop()]
    return None
