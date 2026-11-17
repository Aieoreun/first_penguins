"""v7 흐름. v6과 구조가 같고, 차단 확인 프롬프트만 과목별로 다르다.

검색 1위 섹션의 과목(PY, PR, FIN)에 따라 그 과목의 목차와 판정 기준을 덧붙인 프롬프트로 차단 확인을 부른다.
공개 400문항에서 차단을 놓치는 것이 평균 10건에서 5건으로 줄었다 (docs/sw/04_실험기록.md 16절).
"""
from pipelines.v6.pipeline import Pipeline as V6Pipeline
from pipelines.v7 import prompts


class Pipeline(V6Pipeline):
    async def _ask(self, state, name, system, user, effort, timeout):
        if name == "block" and state.chunks:
            subject = state.chunks[0].section_id.split("-")[0]
            system = prompts.BLOCK_SYSTEMS.get(subject, system)
        return await super()._ask(state, name, system, user, effort, timeout)
