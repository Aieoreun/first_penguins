"""Read an exact KB excerpt before composing an ordinary educational answer."""
import json

from core.schema import Label
from pipelines.v1.pipeline import SECOND_TIMEOUT, _SECTION_MARK, _json_object
from .concise_answer import Fitted


EVIDENCE_INSTRUCTION = """

이번 호출의 출력은 JSON 객체 하나예요. 먼저 질문에 답하는 교재 원문을 찾고,
그 원문의 조건·숫자·이름·행동을 빠뜨리지 않도록 학생에게 보낼 답변을 작성하세요.
교재의 원문을 그대로 복사한 quotes 배열을 먼저 쓰고, answer 문자열을 그다음에 쓰세요.
질문과 직접 관련된 문장이나 표의 행만 고르세요. quotes에는 해설이나 추측을 넣지 마세요.
교재에서 직접 관련된 부분을 찾지 못했으면 quotes는 빈 배열로 쓰고, answer에는
확인할 수 없는 점을 정직하게 말하세요. 답변에 근거 없는 세부 사항을 덧붙이지 마세요.
출력 형식: {"quotes":["교재의 정확한 부분 문자열"],"answer":"학생에게 보낼 답변"}
"""


class Pipeline(Fitted):
    evidence_first = True

    async def _generate(self, state):
        if state.label not in {Label.NORMAL, Label.PII_DISCLOSURE} or not state.section:
            return await super()._generate(state)
        material = self.searcher.section_text(state.section)
        try:
            raw = await self.llm.chat(
                self.answer_systems[state.label] + EVIDENCE_INSTRUCTION,
                json.dumps({"material": material, "conversation": state.messages}, ensure_ascii=False),
                effort=self.answer_effort, max_tokens=self.answer_max_tokens,
                timeout=SECOND_TIMEOUT, json_mode=True)
            obj = _json_object(raw) or {}
            quotes, answer = obj.get("quotes"), obj.get("answer")
            if not isinstance(quotes, list) or not all(isinstance(q, str) and q.strip() and q in material for q in quotes):
                raise ValueError("Invalid KB excerpt")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Missing answer")
            answer = _SECTION_MARK.sub("", answer).strip()
            state.answer = f"{answer} ({state.section})"
            state.citations = [state.section]
        except Exception as exc:
            # The outer request deadline still bounds a fallback generation.
            state.errors.append(f"evidence_answer:{type(exc).__name__}")
            await super()._generate(state)
