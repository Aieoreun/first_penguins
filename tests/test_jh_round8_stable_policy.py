import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
from pipelines import load
load('jh_r7_binary_guard')
from core.schema import Label
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round8_stable_policy import FixedHarmful,BriefHarmful


class StablePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_harmful_tutor_response_without_GPT_or_input_echo(self):
        for cls in [FixedHarmful,BriefHarmful]:
            p=cls.__new__(cls)
            for label in [Label.HARMFUL_REQUEST,Label.HARMFUL_SPEECH]:
                s=p._new_state({'messages':[{'role':'user','content':'사용자 입력을 그대로 복사하지 마세요'}]},False)
                s.label=label
                with patch.object(BinaryGuard,'_generate',new_callable=AsyncMock) as base:
                    await p._generate(s)
                    base.assert_not_awaited()
                response=s.to_tutor_response()
                self.assertEqual(response['answer'],cls.harmful_policy_text[label])
                self.assertEqual(response['citations'],[])
                self.assertEqual(response['action'],'refuse' if label is Label.HARMFUL_REQUEST else 'redirect')
                self.assertNotIn('사용자 입력',response['answer'])

    async def test_guard_and_other_labels_delegate(self):
        p=BriefHarmful.__new__(BriefHarmful)
        for guard,label in [(True,Label.HARMFUL_REQUEST),(False,Label.NORMAL),(False,Label.JAILBREAK_INJECTION),(False,Label.EMOTIONAL_DISTRESS)]:
            s=SimpleNamespace(_material_budget_guard=guard,label=label)
            with patch.object(BinaryGuard,'_generate',new_callable=AsyncMock) as base:
                await p._generate(s);base.assert_awaited_once_with(s)
        self.assertIs(load('jh_r8_brief_harmful'),BriefHarmful)
        self.assertIs(load('jh_r8_fixed_harmful'),FixedHarmful)
        self.assertIs(BriefHarmful._classify,BinaryGuard._classify)
        self.assertIs(BriefHarmful.guard,BinaryGuard.guard)


if __name__=='__main__':unittest.main()
