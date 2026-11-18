"""v9에서 바꾼 것. 나머지는 v8과 같다.

교재 확인 프롬프트에서 근거 섹션을 적는 부분을 뺐다. 근거 섹션은 낱말 검색 1위를 쓰기 때문에(02_설계.md 결정 14)
모델이 적은 섹션 ID는 쓰이지 않는다. 적을 것이 줄어 응답을 못 읽는 실패가 줄었다 (docs/sw/05_로컬_점수표.md).
"""
from pipelines.v2 import prompts as v2


def _without_section(system: str) -> str:
    start = "4. 근거 섹션(section):"
    assert system.count(start) == 1
    head, rest = system.split(start, 1)
    system = head.rstrip("\n") + "\n" + rest.split("\n", 1)[1]
    field = '"section": "<섹션 ID>", '
    assert system.count(field) == 1
    return system.replace(field, "")


MATERIAL_SYSTEM = _without_section(v2.MATERIAL_SYSTEM)
