"""Prompt/payload/MRO contracts; scheduling is tested on the shared fusion."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from core.schema import Label
from pipelines.kimjaehak.peace_conditioned_safety import (
    ConditionedSafety, RequestedConditioned, FALLIBLE_PRIOR,
)
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.peace_guard_variants import RequestedAction
from pipelines.kimjaehak.precise_output import POLICY_HINTS, POLICY_SYSTEM


class ConditionedSafetyTests(unittest.IsolatedAsyncioTestCase):
    def case(self, cls=ConditionedSafety, label=Label.NORMAL, reliable=True, guard=True):
        pipeline = cls.__new__(cls)
        pipeline.llm = SimpleNamespace(chat=AsyncMock(return_value='{"safe":true}'))
        state = pipeline._new_state({'messages': [{'role': 'user', 'content': '설명해 주세요'}],
                                      'candidate_answer': '검사할 후보 답변'}, guard)
        state.label = label
        state.debug['fusion_input_reliable'] = reliable
        return pipeline, state

    async def test_real_safety_call_uses_completed_prediction_and_conditional_hint(self):
        for cls in (ConditionedSafety, RequestedConditioned):
            p, state = self.case(cls, Label.OFF_TOPIC)
            before = deepcopy(state.__dict__)
            await p._fusion_safety(state)
            p.llm.chat.assert_awaited_once()
            call = p.llm.chat.await_args
            system, user = call.args
            self.assertIn(POLICY_SYSTEM, system)
            self.assertIn(FALLIBLE_PRIOR, system)
            self.assertIn(POLICY_HINTS['off_topic'], system)
            for category, hint in POLICY_HINTS.items():
                if category != 'off_topic':
                    self.assertNotIn(hint, system)
            self.assertEqual(json.loads(user), {
                'conversation': state.messages, 'candidate_answer': state.candidate_answer,
                'input_assessment': {'category': 'off_topic', 'decision': 'block'}})
            self.assertEqual(call.kwargs, {'effort': 'low', 'max_tokens': 1800,
                                           'timeout': 13.0, 'json_mode': True})
            self.assertEqual(state.__dict__, before)

    async def test_unreliable_or_missing_label_exposes_no_category_hint(self):
        for mode in ('false', 'missing_flag', 'missing_label', 'truthy_string'):
            p, state = self.case(label=Label.JAILBREAK_INJECTION)
            if mode == 'false':
                state.debug['fusion_input_reliable'] = False
            elif mode == 'missing_flag':
                del state.debug['fusion_input_reliable']
            elif mode == 'missing_label':
                state.label = None
            else:
                state.debug['fusion_input_reliable'] = 'true'
            await p._fusion_safety(state)
            system, user = p.llm.chat.await_args.args
            self.assertIsNone(json.loads(user)['input_assessment'])
            self.assertTrue(all(hint not in system for hint in POLICY_HINTS.values()))
            self.assertEqual(p.llm.chat.await_count, 1)

    async def test_model_verdict_is_not_copied_from_input_allow_or_block(self):
        for label, raw, expected in [(Label.NORMAL, '{"safe":false}', False),
                                     (Label.HARMFUL_REQUEST, '{"safe":true}', True)]:
            p, state = self.case(label=label)
            p.llm.chat.return_value = raw
            await p._fusion_safety(state)
            self.assertIs(state.safe, expected)
            self.assertEqual(state.errors, [])
            self.assertEqual(p.llm.chat.await_count, 1)
            self.assertEqual(json.loads(p.llm.chat.await_args.args[1])['input_assessment'],
                             {'category': state.category.value, 'decision': state.decision.value})

    async def test_strict_boolean_failures_and_cancellation_stay_inherited(self):
        for raw in ('{"safe":"true"}', '{"safe":1}', '{"safe":null}', '{}'):
            p, state = self.case()
            p.llm.chat.return_value = raw
            await p._fusion_safety(state)
            self.assertFalse(state.safe)
            self.assertEqual(state.errors, ['independent_safety:ValueError'])
            self.assertEqual(p.llm.chat.await_count, 1)
        p, state = self.case()
        p.llm.chat.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await p._fusion_safety(state)
        self.assertEqual(state.errors, [])
        self.assertEqual(p.llm.chat.await_count, 1)

    async def test_concurrent_calls_do_not_swap_hints_or_mutate_payload_state(self):
        p, normal = self.case()
        _, blocked = self.case(label=Label.ACADEMIC_DISHONESTY)
        payloads = []
        ready = asyncio.Event()
        async def chat(system, user, **kwargs):
            payload = json.loads(user)
            payloads.append((system, payload))
            if len(payloads) == 2:
                ready.set()
            await ready.wait()
            return '{"safe":true}'
        p.llm.chat.side_effect = chat
        original_messages = [deepcopy(s.messages) for s in (normal, blocked)]
        await asyncio.wait_for(asyncio.gather(p._fusion_safety(normal), p._fusion_safety(blocked)), 1)
        self.assertEqual(p.llm.chat.await_count, 2)
        for system, payload in payloads:
            if payload['input_assessment']['category'] == 'normal':
                self.assertTrue(all(hint not in system for hint in POLICY_HINTS.values()))
            else:
                self.assertIn(POLICY_HINTS['academic_dishonesty'], system)
        self.assertEqual([s.messages for s in (normal, blocked)], original_messages)
        self.assertNotIn('input_assessment', normal.debug)
        self.assertNotIn('input_assessment', blocked.debug)

    def test_requested_composition_changes_only_guard_block_prompt(self):
        self.assertEqual(ConditionedSafety.guard_safety_system, POLICY_SYSTEM + '\n' + FALLIBLE_PRIOR)
        self.assertEqual(ConditionedSafety.guard_safety_hint_systems, POLICY_HINTS)
        self.assertIsNot(ConditionedSafety.guard_safety_hint_systems, POLICY_HINTS)
        p, guard = self.case(RequestedConditioned, guard=True)
        _, tutor = self.case(RequestedConditioned, guard=False)
        self.assertEqual(p._block_system(guard), RequestedAction.guard_block_system)
        baseline = Fusion.__new__(Fusion)
        self.assertEqual(p._block_system(tutor), baseline._block_system(tutor))
        self.assertIs(RequestedConditioned._block_system, RequestedAction._block_system)
        self.assertIs(RequestedConditioned._fusion_safety_payload, ConditionedSafety._fusion_safety_payload)
        self.assertIs(RequestedConditioned._fusion_safety_system, ConditionedSafety._fusion_safety_system)
        for cls in (ConditionedSafety, RequestedConditioned):
            self.assertTrue(cls.guard_safety_after_input)
            for method in ('tutor', '_fusion_safety', '_fusion_ground', '_classify', '_generate'):
                self.assertIs(getattr(cls, method), getattr(Fusion, method))
            for name in ('guard_deadline', 'guard_safety_timeout', 'guard_prepare_limit', 'output_votes'):
                self.assertEqual(getattr(cls, name), getattr(Fusion, name))


if __name__ == '__main__':
    unittest.main()
