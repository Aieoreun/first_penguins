"""Opt-in numeric tutor explanations with explicit input-value provenance."""
import re
from core.schema import Label
from .round7_tax_scope import TaxScopedConcrete
from .round7_binary_guard import BinaryGuard

VALUE_RULE = """
# 학생의 수치와 교재 예시를 구별
마지막 질문이 바꾼 금액·기간·횟수를 먼저 확인하세요. 교재 예시의 값은 계산 규칙을 보여 주는 예일 뿐입니다.
답에는 학생이 실제로 제시한 값과 단위를 사용하세요. 월·연 이율, 회차별 납입 기간, 잔액에서 사용한 금액을 구별하세요.
교재 사실을 말하는 문장과 학생의 조건에 적용한 결과를 구분하세요. 적용 결과는 '질문의 조건으로 계산하면'이라고 밝혀 주세요.
교재의 예시 결과를 다시 말하는 것으로 학생의 계산 질문을 대신하지 마세요.
"""


class ValuePrompt(TaxScopedConcrete):
    tutor_apply_system=TaxScopedConcrete.tutor_apply_system+VALUE_RULE


class ValueAnchored(ValuePrompt):
    tutor_apply_max_chars=600
    tutor_apply_system=ValuePrompt.tutor_apply_system+"""
최종 JSON은 supported, reason, rule_quote, question_values, answer입니다.
supported=true일 때 rule_quote는 계산 규칙을 보여 주는 교재 원문의 연속된 인용 160자 이내,
question_values는 계산에 사용한 학생 제시 수치·단위의 원문 문자열 목록(최대8개, 각40자 이내),
answer는 질문 조건에 적용한 결과와 필요한 계산식만 240자 이내로 작성하세요.
question_values에는 교재 예시의 숫자나 학생의 개인정보를 넣지 마세요.
supported=false일 때 rule_quote와 answer는 빈 문자열, question_values는 빈 목록입니다.
"""

    def _apply_eligible(self,state):
        return state.label==Label.NORMAL and super()._apply_eligible(state)

    def _attach_explanation(self,state,obj):
        if obj.get('supported') is not True:
            return super()._attach_explanation(state,obj)
        rule=obj.get('rule_quote');values=obj.get('question_values');answer=obj.get('answer')
        material=self.searcher.section_text(state.section)
        users=[m.get('content','') for m in state.messages if m.get('role')=='user']
        if (not isinstance(rule,str) or not rule.strip() or len(rule)>160 or rule not in material
                or not isinstance(values,list) or not 1<=len(values)<=8
                or not isinstance(answer,str) or not answer.strip() or len(answer)>240):
            raise ValueError('Invalid grounded numeric explanation')
        if any(not isinstance(v,str) or not 1<=len(v)<=40 or not re.search(r'\d',v)
               or not any(v in text for text in users) for v in values):
            raise ValueError('Question values must be copied from user messages')
        rendered='질문 조건: '+', '.join(values)+'\n교재 규칙: '+rule+'\n적용 결과: '+answer.strip()
        # Source copying is a provenance check, not proof of arithmetic validity.
        super()._attach_explanation(state,{**obj,'answer':rendered})
        state.debug['r8_value_anchored']=True


class BinaryValuePrompt(BinaryGuard,ValuePrompt):
    pass


class BinaryValueAnchored(BinaryGuard,ValueAnchored):
    pass
