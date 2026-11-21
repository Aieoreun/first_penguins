"""v12 흐름. 팀원 흐름의 튜터 답변 방식과 우리의 가벼운 가드를 묶은 것이다.

가져온 것 (김재학의 흐름, pipelines/kimjaehak/extractive_answer.py와 focused_excerpt.py)
  - 정상 질문과 자기 정보가 담긴 질문에는 모델이 답을 쓰지 않고, 근거 섹션의 교재 원문을 그대로 붙여 답한다.
    질문과 가까운 소단원이 앞에 오게 순서만 바꾼다. 답변에 든 사실이 모두 교재 그대로라 지어낼 수가 없다.
    실제 제출에서 이 방식의 튜터 점수가 우리(69~73)보다 높았다(v11을 우리 런박스에서 냈을 때 77).

우리 것을 그대로 둔 것 (v9)
  - 근거 섹션은 낱말 검색 1위다. v11은 모델이 여섯 섹션 중에서 고르는데, 공개 문항에서 인용이 맞은 비율이
    그쪽 0.85, 낱말 검색 1위 0.89~0.91이었다. 호출도 하나 준다.
  - 입력 가드는 세 확인을 한 번씩 부른다. 가드의 출력 검사도 한 번이다.
    v11은 요청당 6~11번을 불러 우리 런박스에서 요청 셋 중 하나에 시간 초과가 났다.

새로 바꾼 것
  - 과제 대행 요구에는 모델이 힌트를 쓰지 않고 고정 문장을 낸다. 모델이 쓴 힌트의 29%가 로컬 채점에서
    "요청에 응했다"로 판정돼 0점이 됐다(02_설계.md 결정 12-3에 적어 둔 위험). 문장은 초안이다.

요청당 모델 호출: 튜터 3번(정답만 요구는 4번), 가드 4번. 근거와 잰 결과는 docs/sw/04_실험기록.md 21절.
"""
from core.schema import ANSWER_MAX_CHARS, Label
from pipelines.v9.pipeline import Pipeline as V9Pipeline

QUOTE_INTRO = "관련 교재 내용을 인용할게요.\n\n"
# 과제 대행 요구에 내는 고정 문장. 초안이다. 이승원이 직접 다듬는다
ACADEMIC_ANSWER = ("제출할 결과물을 대신 완성해 줄 수는 없어요. "
                   "어느 부분에서 막혔는지 알려 주면 그 부분을 스스로 풀 수 있게 같이 살펴볼게요.")
_QUOTED = (Label.NORMAL, Label.PII_DISCLOSURE)


class Pipeline(V9Pipeline):
    quote_material = True    # 정상·자기 정보 질문에 교재 원문을 붙여 답할지
    fixed_academic = True    # 과제 대행 요구에 고정 문장을 낼지

    def _ordered_section(self, state) -> str:
        """근거 섹션 전체. 질문과 가까운 소단원이 앞에 온다. 소단원 안의 글, 코드, 표는 그대로다."""
        chunks = self.searcher.chunks
        indices = [i for i, chunk in enumerate(chunks) if chunk.section_id == state.section]
        if not indices:
            return ""
        full = "\n".join(message["content"] for message in state.messages)
        relevance = {i: 0.0 for i in indices}
        for query in dict.fromkeys((full, state.last_user)):  # 대화 전체와 마지막 질문, 같으면 한 번만
            scores = self.searcher.bm25.scores(query)
            scale = max(scores[i] for i in indices) or 1.0
            for i in indices:
                relevance[i] += scores[i] / scale
        ordered = sorted(indices, key=lambda i: -relevance[i])
        title = chunks[indices[0]].text.split("\n", 1)[0]
        parts = [chunks[i].text.split("\n", 1)[1] if "\n" in chunks[i].text else "" for i in ordered]
        return title + "\n" + "\n\n".join(part for part in parts if part)

    async def _generate(self, state) -> None:
        label = state.label if state.label is not None else Label.NORMAL  # 분류가 실패하면 정상으로 본다
        if self.fixed_academic and label is Label.ACADEMIC_DISHONESTY:
            state.answer, state.citations = ACADEMIC_ANSWER, []
            return
        if self.quote_material and label in _QUOTED and state.section:
            excerpt = self._ordered_section(state).strip()
            answer = QUOTE_INTRO + excerpt
            if excerpt and len(answer) <= ANSWER_MAX_CHARS:
                state.answer, state.citations = answer, [state.section]
                return
        await super()._generate(state)  # 그 밖에는 v9와 같다 (정답만 요구는 생성, 교재 밖은 고정 문장)
