"""Tutor-only non-echoing harmful-policy responses; guard and routing unchanged."""
from core.schema import Label
from .calibrated import FIXED
from .round7_binary_guard import BinaryGuard

BRIEF = {
    Label.HARMFUL_REQUEST: '그 요청에는 응할 수 없어요. 서로의 권리와 안전을 존중하는 범위에서 학습을 도와드릴게요.',
    Label.HARMFUL_SPEECH: '그 주장에는 동의하지 않아요. 사람을 존중하는 표현으로 바꿔서 수업 주제로 이야기해 주세요.',
}


class FixedHarmful(BinaryGuard):
    tutor_harmful_policy='existing_fixed'
    harmful_policy_text=FIXED

    async def _generate(self,state):
        if (getattr(state,'_material_budget_guard',None) is False
                and state.label in self.harmful_policy_text):
            state.answer=self.harmful_policy_text[state.label]
            state.citations=[]
            state.debug['r8_fixed_harmful']=True
            return
        return await super()._generate(state)


class BriefHarmful(FixedHarmful):
    tutor_harmful_policy='brief_no_echo'
    harmful_policy_text=BRIEF
