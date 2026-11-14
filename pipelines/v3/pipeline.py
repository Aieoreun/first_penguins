"""v3 흐름. v2의 구조에, 2026-11-14 실험에서 확인용 문항으로 검증한 설정을 고정한 것이다.

v2와 다른 점 (docs/sw/04_실험기록.md):
  - 입력 가드의 세 확인을 세 번씩 불러 다수결한다 (요청당 확인 호출 9번)
  - 과제 대행과 정답만 요구가 함께 잡히면 정답만 요구로 본다
  - 차단된 입력의 후보 답변은 모델에 묻지 않고 안전하지 않다고 본다
  - 답변 생성과 근거 없는 구절 찾기에 둘째 판 프롬프트를 쓴다
  - 답변 생성의 추론 수준을 low로 올린다
"""
from pipelines.v2 import prompts
from pipelines.v2.pipeline import Pipeline as V2Pipeline


class Pipeline(V2Pipeline):
    votes = 3
    seek_first = True
    safe_by_rule = True
    answer_systems = prompts.ANSWER_SYSTEMS
    unsupported_system = prompts.UNSUPPORTED_SYSTEM
    answer_effort = "low"
