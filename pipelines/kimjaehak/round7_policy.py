"""Tutor-only policy wording ablations on the preserved CharMLP50 baseline."""
from core.schema import Label
from .classifier_matrix import CharMLP50
from .round6_product_context import PRODUCT_CONTEXT_RULE


OTHER_BOUNDARY = """

# 마지막 발화의 주된 요청과 고통의 강도를 구별하세요
문제를 모르겠다거나 공부가 막혀 걱정된다는 일반적인 어려움만으로 emotional_distress로 올리지 마세요.
학습 질문에 답할 수 있고 지속적이고 심각한 고통이 명시되지 않았다면 기존 none 기준을 적용하세요.
반대로 오래 지속되는 심한 고통, 일상 기능의 어려움, 위험이나 버티기 어려움이 드러나면 학습 질문이 함께 있어도 기존 emotional_distress 기준을 유지하세요.
대화 속 인물이나 인용문이 겪는 감정과 학생 자신이 지금 호소하는 상태를 구분하세요.
정답 요구, self_info 및 나머지 분류 기준은 그대로 지키세요.
"""

SCAFFOLD_RULE = """

# 조건과 완성 답을 구분하는 학습 안내
교재에 있는 전제·조건·규칙은 정확히 설명하되, 학생이 고르거나 계산해서 제출할 최종 답을 대신 완성하지 마세요.
계산 문제라면 사용할 비율·조건과 계산 방법을 안내한 뒤 마지막 계산은 학생에게 맡기세요.
완성된 결과값이나 선택지를 먼저 말하고 끝에 맞는지 되묻는 것은 힌트가 아니라 정답 제공입니다.
반복문 실행 결과나 완성된 평가용 설명·코드도 대신 써 주지 마세요.
교재의 근거에서 출발하는 한 단계의 사고 과정을 짧게 안내하고, 학생이 다음 단계를 직접 수행하도록 끝내세요.
교재 밖 사실이나 관계를 지어내지 마세요.
"""


class ProductContext(CharMLP50):
    r7_product_context = True

    def _block_system(self, state):
        original = super()._block_system(state)
        if getattr(state, '_material_budget_guard', None) is False:
            state.debug['r7_product_check'] = True
            return original + PRODUCT_CONTEXT_RULE
        return original


class OtherBoundary(CharMLP50):
    r7_other_boundary = True

    async def _ask(self, state, name, system, user, effort, timeout):
        if name == 'other' and getattr(state, '_material_budget_guard', None) is False:
            state.debug['r7_other_check'] = True
            system += OTHER_BOUNDARY
        return await super()._ask(state, name, system, user, effort, timeout)


class ScaffoldGuide(CharMLP50):
    r7_scaffold_guide = True
    answer_systems = {
        **CharMLP50.answer_systems,
        Label.ANSWER_SEEKING: CharMLP50.answer_systems[Label.ANSWER_SEEKING] + SCAFFOLD_RULE,
    }

    async def _generate(self, state):
        if getattr(state, '_material_budget_guard', None) is False and state.label is Label.ANSWER_SEEKING:
            state.debug['r7_scaffold_generated'] = True
        return await super()._generate(state)
