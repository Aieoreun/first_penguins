"""Let answer generation choose one source from the top three KB sections."""
import json

from core.schema import FALLBACK_ANSWER, Label
from pipelines.v1.pipeline import SECOND_TIMEOUT, _APPEND_ID, _CITE, _SECTION_MARK, _SENTENCE_END, _json_object
from .voted import Pipeline as Voted


class Pipeline(Voted):
    answer_sections = 3

    async def _generate(self, state):
        label = state.label if state.label is not None else Label.NORMAL
        ids = list(dict.fromkeys(c.section_id for c in state.chunks))[:self.answer_sections]
        materials = [{"section_id": sid, "text": self.searcher.section_text(sid)} for sid in ids]
        system = self.answer_systems[label] + """

이번 호출에서는 여러 교재 섹션이 제공돼요. 마지막 질문의 실습·상품·인물을 정확히 구별하고
답변에 가장 직접적으로 쓰는 섹션 하나를 고르세요. 비슷한 이름의 다른 실습을 혼동하지 마세요.
질문에 답이 되는 교재의 핵심 사실을 빠뜨리지 말고, 대상의 이름과 조건도 분명히 밝혀요.
위의 교육적 응답 방식은 answer 문자열 안에 적용해요. 호출의 출력 형식만 다음 JSON으로 바꿔요.
{"section":"제공된 section_id 중 하나", "answer":"학생에게 보낼 답변"}
JSON 밖에는 아무것도 쓰지 마세요. 제공되지 않은 섹션 ID를 만들지 마세요."""
        try:
            text = await self.llm.chat(system, json.dumps({"materials": materials, "conversation": state.messages}, ensure_ascii=False),
                                       effort=self.answer_effort, max_tokens=self.answer_max_tokens,
                                       timeout=SECOND_TIMEOUT, json_mode=True)
            obj = _json_object(text) or {}
            section, answer = obj.get("section"), obj.get("answer")
            if section not in ids or not isinstance(answer, str) or not answer.strip():
                raise ValueError("Invalid answer source or text")
            state.section = section
            answer = _SECTION_MARK.sub("", answer).strip()
            cite = label in _CITE or (label is Label.OUT_OF_MATERIAL and len(_SENTENCE_END.findall(answer)) >= 2)
            if cite and label in _APPEND_ID:
                answer = f"{answer} ({section})"
            state.answer, state.citations = answer, [section] if cite else []
        except Exception as exc:
            state.errors.append(f"generate:{type(exc).__name__}")
            state.answer, state.citations = FALLBACK_ANSWER, []
