"""Keep factual grounding semantics; request the smallest meaningful error span."""
from .classifier_matrix import CharMLP50
from .round7_adaptive_ground import AdaptiveGround

MINIMAL_CLAUSE_RULE="""

# unsupported 구절의 경계
먼저 교재와 대화를 대조해 사실 오류가 있는지 판단하고, 오류가 확정된 부분만 아래 기준으로 짧게 인용하세요.
정상 사실을 오류로 바꾸거나, 짧게 만들기 위해 필요한 오류 조건을 빼면 안 됩니다.
- 숫자·범위·값·명칭이 틀렸다면 무엇의 값인지 이해할 수 있는 가장 작은 구절을 고르세요.
- 앞의 맞는 주제명·주어·도입말·조건과 뒤의 맞는 안내는 오류 구절에 붙이지 마세요.
- 한 문장 안에 맞는 주장과 틀린 주장이 섞였으면 틀린 주장만 분리하세요. 독립적인 오류가 여러 개면 각각 인용하세요.
- 단순한 조사·종결어미·출처 표시를 굳이 붙이지 마세요. 단, 오류의 의미를 완성하는 술어·부정·조건은 보존하세요.
- 후보 답변의 글자를 그대로 복사하세요. 요약·교정문을 쓰거나 여러 위치의 문구를 하나로 이어 붙이지 마세요.
- 틀린 값만 지나치게 짧게 떼어 같은 숫자의 정상 사용까지 지목하지 마세요.
확정된 사실 오류가 없으면 기존처럼 {"unsupported":[]}입니다. 원래 근거 판정 기준과 JSON 형식은 바뀌지 않습니다.
"""

class MinimalClause(CharMLP50):
    guard_minimal_clause=True
    unsupported_system=CharMLP50.unsupported_system+MINIMAL_CLAUSE_RULE

class AdaptiveMinimal(MinimalClause,AdaptiveGround):
    pass
