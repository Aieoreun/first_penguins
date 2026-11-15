"""v4 흐름. v2의 구조(입력 가드를 셋으로 나눠 동시에)에 v3에서 좋았던 설정을 더하고, 다수결은 뺀 것이다.

v3와 같은 점 (docs/sw/04_실험기록.md):
  - 과제 대행과 정답만 요구가 함께 잡히면 정답만 요구로 본다
  - 차단된 입력의 후보 답변은 모델에 묻지 않고 안전하지 않다고 본다
  - 답변 생성과 근거 없는 구절 찾기에 둘째 판 프롬프트를 쓴다
  - 답변 생성의 추론 수준을 low로 올린다

v3와 다른 점:
  - 입력 가드의 세 확인을 한 번씩만 부른다. 요청당 모델 호출이 네 번이다(확인 3 + 답변 또는 출력 가드 1).
    v3는 세 번씩 불러 동시 64개 요청에서 시간 초과가 났다 (04_실험기록.md 8절)
  - 요청 전체에 마감 시간을 둔다. 앞 단계가 늦으면 뒤 단계가 쓸 시간이 그만큼 준다.
    마감을 넘긴 단계는 실패한 것으로 보고, 그 단계가 실패했을 때의 값(02_설계.md 결정 15의 표)을 낸다
"""
import asyncio
import os
import time

from core.schema import FALLBACK_ANSWER
from core.state import State
from pipelines.v2 import prompts
from pipelines.v2.pipeline import Pipeline as V2Pipeline

REQUEST_DEADLINE = float(os.environ.get("V4_DEADLINE", "24"))      # 요청 하나가 쓸 수 있는 시간(초). 규격의 제한은 30초다
PREPARE_LIMIT = float(os.environ.get("V4_PREPARE_LIMIT", "14"))    # 그중 검색과 입력 가드가 쓸 수 있는 시간(초)


class Pipeline(V2Pipeline):
    votes = 1
    seek_first = True
    safe_by_rule = True
    answer_systems = prompts.ANSWER_SYSTEMS
    unsupported_system = prompts.UNSUPPORTED_SYSTEM
    answer_effort = "low"
    deadline = REQUEST_DEADLINE
    prepare_limit = PREPARE_LIMIT

    async def _within(self, coro, seconds: float, state: State, name: str) -> bool:
        """coro를 seconds 안에 끝낸다. 못 끝내면 멈추고 False를 돌려준다."""
        try:
            await asyncio.wait_for(coro, timeout=max(0.05, seconds))
            return True
        except asyncio.TimeoutError:
            state.errors.append(f"{name}:Deadline")
            return False

    async def _prepare_within(self, state: State, extra: str = "") -> None:
        await self._within(self.prepare(state, extra), self.prepare_limit, state, "prepare")
        # 입력 가드가 마감에 걸리면 범주는 normal로 남는다. 근거 섹션은 낱말 검색 1위로 채운다 (결정 14)
        if state.section is None and state.chunks:
            state.section = state.chunks[0].section_id

    async def tutor(self, body) -> State:
        started = time.monotonic()
        state = State.from_request(body)
        await self._prepare_within(state)
        if state.needs_generation:
            left = self.deadline - (time.monotonic() - started)
            if not await self._within(self._generate(state), left, state, "generate"):
                state.answer, state.citations = FALLBACK_ANSWER, []  # 답변 생성 실패와 같다
        return state

    async def guard(self, body) -> State:
        started = time.monotonic()
        state = State.from_request(body)
        await self._prepare_within(state, extra=state.candidate_answer or "")
        if state.has_candidate:
            left = self.deadline - (time.monotonic() - started)
            if not await self._within(self._check_output(state), left, state, "check_output"):
                state.grounded, state.unsupported = True, []  # 근거 검사 실패와 같다. safe는 이미 정해져 있다
        return state
