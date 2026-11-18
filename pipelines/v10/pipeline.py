"""v10 흐름. v9와 같고, 낮은 확신으로 잡은 판정을 세 범주에서 버린다.

차단 확인과 그 밖 확인은 판정과 함께 확신 정도(0~100)를 낸다. 탈옥, 과제 대행, 정답만 요구는 확신이 85보다 낮으면
그 판정을 없던 것으로 한다("해당 없음"). 유해, 주제 밖, 정서적 고통, 뜻 모를 입력은 확신이 낮아도 그대로 둔다.

근거 (공개 400문항 전체 채점 아홉 번을 합친 값, docs/sw/HANDOFF.md "실험 아이디어"):
  - 확신 85 미만으로 잡은 판정이 맞은 수/틀린 수: 정답만 요구 9/116, 탈옥 17/46, 과제 대행 10/24
  - 나머지 범주는 확신이 낮아도 대부분 맞는다 (뜻 모를 입력은 80 미만에서도 29/4)
알려진 대가: 차단을 놓치는 것이 조금 는다(가드 문항 평균 3건 → 4.4건). 그 안에 탈옥이 들어온다.
로컬 전체 채점으로는 이득이 흔들림보다 작아 확인되지 않았다 (docs/sw/05_로컬_점수표.md).
"""
import os

from pipelines.v9.pipeline import Pipeline as V9Pipeline

MIN_CONFIDENCE = float(os.environ.get("V10_MIN_CONFIDENCE", "85"))
DROP_WHEN_UNSURE = frozenset({"jailbreak_injection", "academic_dishonesty", "answer_seeking"})


class Pipeline(V9Pipeline):
    min_confidence = MIN_CONFIDENCE
    drop_when_unsure = DROP_WHEN_UNSURE

    async def _ask(self, state, name, system, user, effort, timeout):
        answer = await super()._ask(state, name, system, user, effort, timeout)
        if name in ("block", "other"):
            confidence = answer.get("confidence")
            if (answer.get("category") in self.drop_when_unsure
                    and isinstance(confidence, (int, float)) and confidence < self.min_confidence):
                answer = dict(answer, category="none")
        return answer
