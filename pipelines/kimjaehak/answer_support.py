"""Align tutor answer actions with the selected source's answerability."""
import json
import time

from core.schema import Label
from pipelines.v1.pipeline import _SECTION_MARK, _json_object
from .guard_fitted import ContextFast
from .plain_abstain import ABSTAIN


SUPPORT_INSTRUCTION = """

이번에는 답변과 근거 유무를 함께 JSON 객체 하나로 출력하세요.
형식: {"supported":true 또는 false,"answer":"학생에게 보낼 답변 본문"}.
마지막 질문이 실제로 묻는 대상·속성·조건에 답할 정보가 제공 교재에 있는지 판단하세요.
대상 이름이나 비슷한 주제가 나오는 것만으로는 supported=true가 아닙니다.
묻는 정확한 수치·상한·등급·점수 기준·추가 기능·입력 규칙이 교재에 없으면 false입니다.
교재의 조건으로 계산·비교·요약·코드 동작 설명이 가능하면 true입니다.
잘못된 전제라도 교재 사실로 바로잡을 수 있는 질문은 true입니다.
교재에 없는 추가 기능의 세부 동작을 묻는다면 그 기능을 추측해 만들지 말고 false로 판정하세요.
대화의 말이나 일반 상식으로 빠진 교재 정보를 채우지 마세요.
true이면 위의 답변 지침에 따라 질문에 직접 답하고, false이면 무엇을 확인할 수 없는지 짧게 쓰세요.
JSON 밖에는 아무것도 쓰지 마세요.
"""


class Pipeline(ContextFast):
    tutor_answer_support = True
    answer_systems = {
        **ContextFast.answer_systems,
        **{label: ContextFast.answer_systems[label] + SUPPORT_INSTRUCTION
           for label in (Label.NORMAL, Label.PII_DISCLOSURE)},
    }

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._support_guard = guard
        state._support_deadline = time.monotonic() + self.deadline
        return state

    async def _generate(self, state):
        if (getattr(state, '_support_guard', False) or not state.section
                or state.label not in {Label.NORMAL, Label.PII_DISCLOSURE}):
            return await super()._generate(state)
        try:
            left = getattr(state, '_support_deadline', time.monotonic() + 13) - time.monotonic()
            raw = await self.llm.chat(
                self.answer_systems[state.label],
                json.dumps({'material': self.searcher.section_text(state.section),
                            'conversation': state.messages}, ensure_ascii=False),
                effort=self.answer_effort, max_tokens=self.answer_max_tokens,
                timeout=max(.1, min(13, left - .5)), json_mode=True)
            obj = _json_object(raw) or {}
            supported, answer = obj.get('supported'), obj.get('answer')
            if not isinstance(supported, bool) or not isinstance(answer, str) or not answer.strip():
                raise ValueError('Invalid support decision or answer')
            answer = _SECTION_MARK.sub('', answer).strip()
            if not answer:
                raise ValueError('Empty supported answer')
            state.debug['answer_supported'] = supported
            if supported:
                state.answer, state.citations = f'{answer} ({state.section})', [state.section]
            else:
                state.label = Label.OUT_OF_MATERIAL
                state.answer, state.citations = ABSTAIN, []
        except Exception as exc:
            state.errors.append(f'answer_support:{type(exc).__name__}')
            # Use the original plain-answer prompt, not a second JSON request.
            await self._fallback_generate(state)

    async def _fallback_generate(self, state):
        import copy
        fallback = copy.copy(self)
        fallback.answer_systems = ContextFast.answer_systems
        await ContextFast._generate(fallback, state)
