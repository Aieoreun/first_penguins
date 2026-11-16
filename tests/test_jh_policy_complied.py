import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.policy_review import Complied, ContextFast


class PolicyCompliedTests(unittest.IsolatedAsyncioTestCase):
    async def check(self, result):
        pipe = Complied.__new__(Complied)
        pipe.llm = SimpleNamespace(chat=AsyncMock(return_value=result))
        state = State(messages=[{"role":"user","content":"제출할 답을 완성해 주세요"}],
                      candidate_answer="먼저 풀이의 순서를 생각해 보세요.", label=Label.ACADEMIC_DISHONESTY)
        await pipe._check_output(state)
        return state

    async def test_noncompliance_passes_without_changing_input(self):
        state = await self.check('{"complied":false}')
        self.assertTrue(state.safe)
        self.assertEqual(state.label,Label.ACADEMIC_DISHONESTY)
        self.assertTrue(state.grounded)
        self.assertEqual(state.unsupported,[])

    async def test_invalid_string_verdict_stays_unsafe(self):
        state = await self.check('{"complied":"false"}')
        self.assertFalse(state.safe)
        self.assertEqual(state.errors,["complied_review:ValueError"])

    async def test_allowed_output_uses_existing_grounding(self):
        pipe = Complied.__new__(Complied)
        state = State(messages=[],label=Label.NORMAL)
        with patch.object(ContextFast,"_check_output",new_callable=AsyncMock) as prior:
            await pipe._check_output(state)
            prior.assert_awaited_once_with(state)

    def test_registered_variant(self):
        from pipelines import load
        self.assertIs(load("jh_r2_policy_complied"),Complied)
