"""v6 흐름. v5와 구조가 같고 튜터 답변 두 가지만 다르다.

  - 정상 답변을 한두 문장으로 줄였다
  - 교재에 없는 질문은 모델을 부르지 않고 고정 문장을 낸다 (인용 없음)
그리고 제한 시간을 바꿨다 (튜터와 가드 모두).
  - 교재 확인 호출: 10초 → 13초. 동시 64개에서 이 호출의 8%가 10초를 넘고 13초를 넘는 것은 0.8%였다
  - 검색과 입력 가드 전체: 14초 → 14.5초, 요청 전체: 24초 → 26초 (규격의 제한은 30초)
이유와 잰 결과는 docs/sw/04_실험기록.md 13절.
"""
import os

from core.schema import Label
from pipelines.v5.pipeline import Pipeline as V5Pipeline
from pipelines.v6 import prompts


class Pipeline(V5Pipeline):
    answer_systems = prompts.ANSWER_SYSTEMS
    fixed_abstain = True  # 교재에 없는 질문에 고정 문장을 낼지
    material_timeout = float(os.environ.get("V6_MATERIAL_TIMEOUT", "13"))
    prepare_limit = float(os.environ.get("V6_PREPARE_LIMIT", "14.5"))
    deadline = float(os.environ.get("V6_DEADLINE", "26"))

    async def _generate(self, state) -> None:
        if self.fixed_abstain and state.label is Label.OUT_OF_MATERIAL:
            state.answer, state.citations = prompts.ABSTAIN_ANSWER, []
            return
        await super()._generate(state)
