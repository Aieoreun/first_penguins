"""Tutor-only support for facts explicitly derivable from provided code/tables."""
from .material_budget import Pipeline as MaterialBudget


COMPUTABLE_SUPPORT = """

# 코드와 표에 들어 있는 근거
교재 근거는 산문 문장뿐 아니라 코드, 표, 수식에도 있을 수 있습니다.
질문이 교재 코드의 동작·출력·반복 횟수나 표의 값으로 계산하는 결과를 묻는다면,
필요한 코드·값·조건이 검색 결과와 질문에 모두 있는지 확인하세요.
그 조건으로 결과를 결정할 수 있으면 결과 문장이 교재에 따로 쓰여 있지 않아도
관련 코드 줄 또는 값과 규칙을 quote에 원문 그대로 옮기세요. 계산한 결과를 quote로 만들지 마세요.
입력값, 실행 조건, 수수료, 외부 데이터 등이 빠져 결과를 결정할 수 없다면 근거가 있다고 추측하지 마세요.
교재에 없는 함수·상품·기능·규칙을 새로 만들거나 상식으로 빈틈을 메우지 마세요.
"""


class Computable(MaterialBudget):
    tutor_computable_material = True

    async def _ask(self, state, name, system, user, effort, timeout):
        if name == 'material' and not getattr(state, '_material_budget_guard', True):
            system += COMPUTABLE_SUPPORT
        return await super()._ask(state, name, system, user, effort, timeout)
