"""Keep all retrieved material, but request only the quote used for routing."""
from pipelines.v1.pipeline import _json_object
from pipelines.v2 import prompts
from .adaptive import Pipeline as Adaptive


MATERIAL_SYSTEM = prompts.MATERIAL_SYSTEM.split("# 출력", 1)[0] + """

# 출력
물음에 답하는 근거가 있으면 필요한 핵심 구절 하나만 그대로 옮긴다. 가능하면 120자 이내로 줄이되 숫자·조건을 바꾸지 않는다.
그 사실이 없으면 빈 문자열이다. asked, section, confidence, 설명은 출력하지 않는다.
아래 JSON 하나만 낸다.
{"quote": "교재의 핵심 구절 또는 빈 문자열"}
"""


class Pipeline(Adaptive):
    material_max_tokens = 2000

    async def _ask(self, state, name, system, user, effort, timeout):
        if name != "material":
            return await super()._ask(state, name, system, user, effort, timeout)
        try:
            text = await self.llm.chat(MATERIAL_SYSTEM, user, effort=effort,
                max_tokens=self.material_max_tokens, timeout=timeout, json_mode=True)
            obj = _json_object(text)
            if obj is None or not isinstance(obj.get("quote"), str):
                state.errors.append("material:parse")
                return {}
            return obj
        except Exception as exc:
            state.errors.append(f"material:{type(exc).__name__}")
            return {}
