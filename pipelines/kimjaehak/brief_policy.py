"""Use short, explicit refusals and respectful redirection for fixed policies."""
from core.schema import Label
from .calibrated import ResponseState
from .candidate import Pipeline as Candidate


POLICY_TEXT = {
    Label.HARMFUL_REQUEST: "요청하신 일은 도와드릴 수 없어요. 다른 사람의 권리와 안전을 지키는 방법은 함께 배울 수 있어요.",
    Label.HARMFUL_SPEECH: "그런 모욕이나 차별에는 동의하지 않아요. 다른 사람을 존중하는 표현으로 바꿔 주세요.",
    Label.OFF_TOPIC: "그 요청은 수업 범위를 벗어나서 도와드릴 수 없어요. 수업과 관련된 질문을 해 주세요.",
    Label.JAILBREAK_INJECTION: "그 요청은 거절할게요. 비공개 지시를 보여 주거나 기존 안전 규칙을 무시할 수는 없어요.",
}


class BriefState(ResponseState):
    def to_tutor_response(self):
        response = super().to_tutor_response()
        if self.label in POLICY_TEXT:
            response["answer"] = POLICY_TEXT[self.label]
        return response


class Pipeline(Candidate):
    async def tutor(self, body):
        state = await super().tutor(body)
        return BriefState(**vars(state))


class ClearScaffold(Pipeline):
    fixed_academic_scaffold = True

    async def _generate(self, state):
        if state.label is Label.ACADEMIC_DISHONESTY:
            state.answer = ("제출할 답안을 대신 완성해 드릴 수는 없어요. 먼저 자료에서 필요한 개념을 찾고, "
                            "문제의 조건과 풀이 순서를 직접 정리해 보세요. 작성한 풀이를 보내주시면 "
                            "어떤 부분을 고치면 좋을지 함께 살펴볼게요.")
            state.citations = []
            return
        await super()._generate(state)
