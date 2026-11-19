"""Separate unprovided attributes from a negative fact or a simplifying premise."""
from core.schema import Label
from .classifier_matrix import CharMLP50
from .plain_abstain import ABSTAIN
from .round7_answer import ConcreteApply


MATERIAL_SCOPE_RULE="""

# 질문의 속성과 교재의 적용 범위를 정확히 맞추세요
- 같은 상품이나 실습의 관련 설명이 있다고 해서 질문한 세부 속성까지 제공된 것은 아닙니다.
- 자료가 어떤 요소를 계산에서 제외하거나 다루지 않는다고 한 것은 그 요소의 실제 값이 0, 무료, 없음이라고 정한 뜻이 아닙니다. 단순화한 연습의 조건과 실제 적용 조건을 구별하세요.
- 질문이 제외된 요소의 실제 값·정책을 요구하면 교재에 그 값이나 규칙이 별도로 있어야 합니다. 없으면 주변 사실을 대신 답할 근거로 삼지 마세요.
- 반대로 질문 자체가 교재의 단순화 가정으로 계산하라고 했거나, 교재가 특정 기능을 포함하지 않는다고 명시했고 바로 그 포함 여부를 묻는다면 답할 수 있습니다.
- 어떤 기능을 다루지 않는다는 사실만으로 그 기능을 나중에 구현했을 때의 세부 동작을 추측하지 마세요.
- 질문의 수치에 교재의 명시된 수식·비율·절차를 적용해 얻을 수 있는 결과는 교재 예시와 숫자가 다르다는 이유만으로 배제하지 마세요.
"""


class MaterialScope(CharMLP50):
    tutor_material_scope=True

    async def _ask(self,state,name,system,user,effort,timeout):
        if name=='material' and getattr(state,'_material_budget_guard',None) is False:
            state._r7_material_scope_used=True
            system+=MATERIAL_SCOPE_RULE
        return await super()._ask(state,name,system,user,effort,timeout)

    async def _classify(self,state):
        try:return await super()._classify(state)
        finally:
            if getattr(state,'_r7_material_scope_used',False):state.debug['r7_material_scope_used']=True


class ScopedConcrete(ConcreteApply):
    tutor_apply_missing_material_abstain=True
    tutor_apply_system=ConcreteApply.tutor_apply_system+MATERIAL_SCOPE_RULE+"""
supported=false일 때 reason도 적으세요. 교재에 필수 사실·규칙이 없으면 missing_material,
학생이 제시해야 할 값이나 조건이 부족하면 missing_question_values, 안전하게 답할 수 없으면 unsafe입니다.
supported=true일 때 reason은 빈 문자열입니다. JSON은 supported, reason, answer 세 필드입니다.
"""

    def _attach_explanation(self,state,obj):
        if (obj.get('supported') is False and obj.get('reason')=='missing_material'
                and isinstance(obj.get('answer'),str)):
            state.label=Label.OUT_OF_MATERIAL
            state.answer,state.citations=ABSTAIN,[]
            state.debug['r7_apply_abstained']=True
            return
        return super()._attach_explanation(state,obj)
