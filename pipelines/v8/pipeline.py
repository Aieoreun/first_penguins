"""v8 흐름. v6과 같고, 출력 가드가 잡은 근거 없는 구절을 코드로 짧게 다듬는다.

모델이 베껴 온 구절은 틀린 자리는 맞게 짚지만 정답보다 조금 길다(정답 중앙 17자, 예측 27자).
앞에 붙은 주어("…은/는")와 끝에 붙은 "이에요/예요/이므로"를 떼어 낸다 (docs/sw/04_실험기록.md 18절).
과목별 차단 확인(v7)은 넣지 않았다. 실제 제출에서 가드가 가장 낮게 나왔다.
"""
import re

from pipelines.v6.pipeline import Pipeline as V6Pipeline

_TAIL = re.compile(r"[\s.,!?]+$")
_COPULA = re.compile(r"(이에요|예요|이예요|입니다|이므로|이라서|이고|이며|이죠|이야|이랍니다)$")
_TOPIC = re.compile(r"^[^\s]+(?:\s[^\s]+){0,3}?[은는]\s+")
MIN_LEFT = 4  # 주어를 떼고 남는 글자가 이보다 적으면 떼지 않는다


def trim(phrase: str) -> str:
    """구절의 앞뒤에서 틀린 값이 아닌 부분을 떼어 낸다. 떼어 낸 결과는 원래 구절의 일부라 후보 답변에 그대로 있다."""
    text = _COPULA.sub("", _TAIL.sub("", phrase)).rstrip()
    match = _TOPIC.match(text)
    if match and len(text) - match.end() >= MIN_LEFT:
        text = text[match.end():]
    return text or phrase


class Pipeline(V6Pipeline):
    trim_spans = True

    async def _check_output(self, state) -> None:
        await super()._check_output(state)
        if self.trim_spans and state.unsupported:
            state.debug = dict(state.debug or {}, raw_unsupported=list(state.unsupported))  # 다듬기 전 구절 (비교용)
            state.unsupported = [trim(p) for p in state.unsupported]
