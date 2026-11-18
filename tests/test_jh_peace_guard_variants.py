import asyncio
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from core.schema import Label
from pipelines.v2 import prompts
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.peace_guard_prompt import PEACE_BLOCK_SYSTEM
from pipelines.kimjaehak.peace_guard_variants import RequestedAction


class RequestedActionTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, cls=RequestedAction):
        p = cls.__new__(cls)
        async def chat(*args, **kwargs):
            await asyncio.sleep(0)
            return '{"requested_action":"explain the example","category":"none","confidence":97}'
        p.llm = SimpleNamespace(chat=AsyncMock(side_effect=chat))
        return p

    def state(self, p, guard):
        return p._new_state({'messages': [{'role': 'user', 'content': 'question'}]}, guard)

    async def ask(self, p, state, name='block'):
        return await p._ask(state, name, 'caller system', 'unchanged payload', 'low', 7)

    def test_exact_effective_peace_prompt_hash(self):
        self.assertEqual(hashlib.sha256(PEACE_BLOCK_SYSTEM.encode()).hexdigest(),
            '2bd470807c6ddaa1d87442f2b1f809398d986d72094c89735e51aa69206a709f')
        self.assertEqual(len(PEACE_BLOCK_SYSTEM), 3991)
        self.assertIn('"requested_action"', PEACE_BLOCK_SYSTEM)
        self.assertTrue(PEACE_BLOCK_SYSTEM.endswith(BLOCK_SYSTEM[BLOCK_SYSTEM.index('\n\n# 경계를 다시 확인하세요'):]))

    async def test_actual_ask_hook_changes_only_guard_block_system(self):
        for guard, name in ((True, 'block'), (False, 'block'), (True, 'other'), (True, 'material')):
            p, base = self.setup_pipeline(), self.setup_pipeline(Fusion)
            s, original = self.state(p, guard), self.state(base, guard)
            changed = await self.ask(p, s, name)
            unchanged = await self.ask(base, original, name)
            self.assertEqual(changed, unchanged)
            self.assertEqual(changed['category'], 'none')
            call, parent_call = p.llm.chat.await_args, base.llm.chat.await_args
            self.assertEqual(call.args[1:], parent_call.args[1:])
            self.assertEqual(call.kwargs, parent_call.kwargs)
            self.assertEqual(call.args[0], PEACE_BLOCK_SYSTEM if guard and name == 'block' else parent_call.args[0])
            self.assertEqual(p.llm.chat.await_count, 1)

    def test_missing_or_non_boolean_endpoint_identity_preserves_parent(self):
        p = self.setup_pipeline()
        s = self.state(p, True)
        del s._material_budget_guard
        self.assertIs(p._block_system(s), BLOCK_SYSTEM)
        for value in (None, False, 'true', 1):
            s._material_budget_guard = value
            self.assertIs(p._block_system(s), BLOCK_SYSTEM)

    async def test_parallel_guard_tutor_calls_do_not_mutate_global_prompt(self):
        p = self.setup_pipeline()
        original_v2, original_calibrated = prompts.BLOCK_SYSTEM, BLOCK_SYSTEM
        guard, tutor = self.state(p, True), self.state(p, False)
        await asyncio.gather(self.ask(p, guard), self.ask(p, tutor))
        self.assertCountEqual([c.args[0] for c in p.llm.chat.call_args_list],
                              [PEACE_BLOCK_SYSTEM, BLOCK_SYSTEM])
        self.assertIs(prompts.BLOCK_SYSTEM, original_v2)
        self.assertIs(Fusion._block_system(p, guard), original_calibrated)
        self.assertIs(p._block_system(tutor), original_calibrated)

    async def test_guard_high_confidence_input_gate_still_skips_block(self):
        p = self.setup_pipeline()
        p.guard_vectorizer = SimpleNamespace(transform=lambda text: text)
        p.guard_classifier = SimpleNamespace(classes_=np.array(['jailbreak_injection', 'normal']),
            predict_proba=lambda text: np.array([[.99, .01]]))
        s = self.state(p, True)
        await p._classify(s)
        self.assertEqual(s.label, Label.JAILBREAK_INJECTION)
        self.assertTrue(s.debug['learned_guard_only'])
        p.llm.chat.assert_not_awaited()

    def test_all_other_methods_models_and_thresholds_are_inherited(self):
        for name in ('__init__', '_classify', '_check_output', '_generate', 'tutor', 'guard',
                     '_collect_checks', '_fusion_ground', '_fusion_safety'):
            self.assertIs(getattr(RequestedAction, name), getattr(Fusion, name))
        for name in ('learned_gate_c', 'guard_only_c', 'learned_gate_threshold',
                     'learned_section_threshold', 'learned_span_threshold', 'deadline',
                     'guard_deadline', 'guard_material_timeout', 'request_slots'):
            self.assertEqual(getattr(RequestedAction, name), getattr(Fusion, name))


if __name__ == '__main__':
    unittest.main()
