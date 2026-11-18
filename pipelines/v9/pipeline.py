"""v9 흐름. v8과 같고, 교재 확인이 근거 섹션을 적지 않는다.

교재 확인이 적는 것이 네 가지(묻는 것, 베낀 문장, 섹션, 확신 정도)에서 세 가지로 준다.
섹션은 쓰이지 않는 값이었다. 판정(베낀 문장이 비었는가)은 그대로다.
로컬 전체 채점에서 교재 확인의 응답을 못 읽는 실패가 5~6건에서 1건으로 줄었고, 교재 안팎 판정의 정확도는 같았다.
"""
from pipelines.v8.pipeline import Pipeline as V8Pipeline
from pipelines.v9 import prompts


class Pipeline(V8Pipeline):
    async def _ask(self, state, name, system, user, effort, timeout):
        if name == "material":
            system = prompts.MATERIAL_SYSTEM
        return await super()._ask(state, name, system, user, effort, timeout)
