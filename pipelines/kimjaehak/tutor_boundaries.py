"""Clarify ordinary learning intent only in the tutor fallback checks."""
from .guard_fitted import ContextFast


OTHER_BOUNDARY = """

# 일상적인 학습 어려움과 심한 고통을 구분하는 절차
emotional_distress의 지속 기간과 심각성은 학생이 실제로 밝힌 내용에서 확인하세요.
'자꾸', '걱정', '막힌다', '어렵다' 같은 단어만으로 장기간의 심한 고통을 추론하지 마세요.
구체적인 학습 질문에 일상적인 어려움이나 걱정을 덧붙인 경우는 none입니다.
반대로 오래 이어진 심한 고통, 일상생활을 유지하기 어려운 상태나 명시적 위기를 밝히면
학습 질문이 함께 있더라도 emotional_distress를 유지하세요. 앞 대화의 실제 진술도 확인하세요.
나머지 answer_seeking, low_signal_input, self_info의 기준은 그대로 적용하세요.
"""

MATERIAL_BOUNDARY = """

# 묻는 사실과 요청한 출력 형식을 구분하세요
학생이 교재 내용을 요약, 표, 암기 카드, 예시 등의 형식으로 바꿔 달라고 할 수 있어요.
이때 그 출력물 자체가 교재에 이미 있는지를 찾지 말고, 만드는 데 필요한 사실·값·동작이 있는지 보세요.
그 사실이 교재에 있으면 해당 근거를 quote에 그대로 적으세요. 요청한 출력 형식의 이름이
교재에 없다는 이유만으로 quote를 비우지 마세요.
반대로 교재에 존재하는 고유 기능·규칙·지수·조건·수치를 묻는다면 그 사실 자체가 있어야 해요.
일반적인 말이나 이름만 관련된 자료로 없는 사실을 있다고 판정하지 마세요.
여러 턴의 대화에서는 마지막 질문의 대상이 앞 대화의 무엇을 가리키는지 먼저 확인하세요.
"""


class BoundaryChecks(ContextFast):
    tutor_other_boundary = False
    tutor_material_boundary = False

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._tutor_boundary_guard = guard
        return state

    async def _ask(self, state, name, system, user, effort, timeout):
        if not getattr(state, '_tutor_boundary_guard', False):
            if name == 'other' and self.tutor_other_boundary:
                system += OTHER_BOUNDARY
            elif name == 'material' and self.tutor_material_boundary:
                system += MATERIAL_BOUNDARY
        return await super()._ask(state, name, system, user, effort, timeout)


class Other(BoundaryChecks):
    tutor_other_boundary = True


class Material(BoundaryChecks):
    tutor_material_boundary = True
