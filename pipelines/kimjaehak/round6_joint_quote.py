"""Tutor joint evidence with lexical quote validation, not semantic proof."""
from .round6_joint_evidence import JOINT_SYSTEM, Pipeline as JointEvidence


QUOTE_SYSTEM = JOINT_SYSTEM.split('reason에는', 1)[0] + """
판정 전에 asked_fact에 현재 질문의 대상(entity)과 묻는 속성·조건(property_or_condition)을 구분해서 적으세요.
quote에는 그 대상의 해당 속성·조건에 직접 답할 근거를 선택 섹션에서 연속된 원문으로 복사하세요.
관련 주제의 설명이나 다른 속성의 근거만으로 질문한 속성이 제공되어 있다고 판단하지 마세요.
어떤 속성이 없거나 확인되지 않는다는 설명은 그 속성의 구체적인 값·규칙을 제공하는 근거가 아니에요.
다만 질문 자체가 그 속성·기능의 존재 여부를 묻는다면, 명시적인 부재 설명도 답할 근거가 됩니다.
계산·비교는 질문의 명시된 조건과 교재의 값·코드·표로 뒷받침해야 하며, quote에는 필요한 원문 근거를 담으세요.
교재 밖 추정이나 대화의 주장으로 빠진 조건을 채우지 마세요. 근거가 없으면 supported=false, quote는 빈 문자열입니다.
asked_fact와 quote를 먼저 확인하고 supported를 결정하세요. 원문에 없는 답을 quote에 쓰지 마세요.
출력 JSON 형식은 다음 하나입니다:
{"asked_fact":{"entity":"질문의 대상","property_or_condition":"묻는 속성 또는 조건"},"quote":"선택 교재의 연속된 원문 또는 빈 문자열","section":"제공된 섹션 ID 하나","supported":true 또는 false}.
"""


class Pipeline(JointEvidence):
    tutor_joint_quote = True
    joint_evidence_system = QUOTE_SYSTEM

    def _adapt_joint_result(self, result, ids):
        # Keep the parent's strict supported/section validation and false path.
        supported, selected = result.get('supported'), result.get('section')
        if not isinstance(supported, bool) or not isinstance(selected, str) or selected not in ids:
            return super()._adapt_joint_result(result, ids)
        if not supported:
            return super()._adapt_joint_result(result, ids)
        quote = result.get('quote')
        compact = ''.join(quote.split()) if isinstance(quote, str) else ''
        text = self.searcher.section_text(selected)
        if not compact or compact not in ''.join(text.split()):
            return {'quote':'', 'section':selected}, None, 'material_joint:InvalidQuote'
        # A substring proves source presence only, not relevance or entailment.
        return {'quote':quote, 'section':selected}, selected, None
