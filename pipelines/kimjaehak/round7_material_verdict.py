"""Tutor-only explicit missing-material verdict inside the existing check call."""
from .classifier_matrix import CharMLP50
from .round7_material_scope import MATERIAL_SCOPE_RULE

VERDICT_RULE="""

# 질문한 사실의 제공 여부를 명시하세요
단순히 관련 문장이 있다는 것과 질문에 필요한 사실이 제공됐다는 것은 다릅니다.
먼저 마지막 질문이 실제로 요구한 속성·조건을 짧은 asked_fact에 쓰고, material_status를 판정하세요.
- sufficient: 그 속성·조건을 교재와 대화의 값으로 설명하거나 계산할 수 있습니다.
- missing_material: 질문이 요구한 교재 고유 사실·규칙이 자료에 없거나 자료가 명시적으로 그 요소를 다루지 않습니다.
- missing_question_values: 교재의 규칙은 있으나 학생이 자신의 값·조건을 더 알려야 합니다. 교재에 규칙이 없는 경우와 구별하세요.
교재의 연습에서 특정 요소를 제외했다면 실제로 그 요소까지 반영한 값을 요구하는 질문은 그 요소의 별도 규칙이 있어야 sufficient입니다.
기존 section과 quote는 유지하되, missing_material이면 quote는 빈 문자열로 두세요.
최종 JSON에는 section, quote, asked_fact, material_status 네 필드를 넣으세요. 학생에게 답변을 생성하지 마세요.
"""


class MaterialVerdict(CharMLP50):
    tutor_material_explicit_status=True

    async def _ask(self,state,name,system,user,effort,timeout):
        enabled=name=='material' and getattr(state,'_material_budget_guard',None) is False
        if enabled:
            system+=MATERIAL_SCOPE_RULE+VERDICT_RULE
            state._r7_material_verdict_used=True
        result=await super()._ask(state,name,system,user,effort,timeout)
        if enabled and result:
            status=result.get('material_status')
            if status=='missing_material':
                state._r7_material_verdict_missing=True
                return {**result,'quote':''}
            if status not in {'sufficient','missing_question_values'}:
                state.errors.append('material_verdict:InvalidStatus')
        return result

    async def _classify(self,state):
        try:return await super()._classify(state)
        finally:
            for suffix in ('used','missing'):
                if getattr(state,'_r7_material_verdict_'+suffix,False):
                    state.debug['r7_material_verdict_'+suffix]=True
