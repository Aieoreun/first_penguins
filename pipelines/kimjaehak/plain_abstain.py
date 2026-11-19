"""Avoid speculative extra facts when the input is outside the supplied material."""
from core.schema import Label
from .learned_sections import TutorOnly


ABSTAIN = "제공된 수업 자료에서는 질문하신 내용을 확인할 수 없어요. 관련 자료를 더 알려 주시면 그 내용을 바탕으로 도와드릴게요."


class Pipeline(TutorOnly):
    async def _generate(self, state):
        if state.label is Label.OUT_OF_MATERIAL:
            state.answer, state.citations = ABSTAIN, []
            return
        await super()._generate(state)
