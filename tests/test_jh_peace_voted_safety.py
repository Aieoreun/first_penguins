"""Offline majority, lifecycle and parent-prompt contracts for safety voting."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.peace_conditioned_safety import ConditionedSafety
from pipelines.kimjaehak.peace_voted_safety import VotedSafety, VotedConditioned
from pipelines.kimjaehak.precise_output import POLICY_HINTS


class VotedSafetyTests(unittest.IsolatedAsyncioTestCase):
    def case(self, cls=VotedSafety):
        p = cls.__new__(cls)
        p.llm = SimpleNamespace(chat=AsyncMock())
        state = p._new_state({'messages': [{'role': 'user', 'content': '설명해 주세요'}],
                              'candidate_answer': '후보 답변'}, True)
        state.label = Label.OFF_TOPIC
        state.debug['fusion_input_reliable'] = True
        return p, state

    async def test_three_parallel_calls_keep_parent_prompt_payload_and_parameters(self):
        for cls, parent in ((VotedSafety, Fusion), (VotedConditioned, ConditionedSafety)):
            p, state = self.case(cls)
            calls, ready = [], asyncio.Event()
            async def chat(system, user, **kwargs):
                index = len(calls)
                calls.append((system, user, kwargs))
                if len(calls) == 3:
                    ready.set()
                await ready.wait()  # A serial implementation would not finish.
                return json.dumps({'safe': index < 2})
            p.llm.chat.side_effect = chat
            expected_payload = parent._fusion_safety_payload(p, state)
            expected_system = parent._fusion_safety_system(p, state)
            await asyncio.wait_for(p._fusion_safety(state), 1)
            self.assertTrue(state.safe)
            self.assertEqual(state.debug['fusion_completed_safety_votes'], 3)
            self.assertEqual(len(calls), 3)
            for system, user, kwargs in calls:
                self.assertEqual(system, expected_system)
                self.assertEqual(json.loads(user), expected_payload)
                self.assertEqual(kwargs, {'effort': 'low', 'max_tokens': 1800,
                                          'timeout': 13.0, 'json_mode': True})
            if cls is VotedConditioned:
                self.assertIn(POLICY_HINTS['off_topic'], expected_system)
                self.assertEqual(expected_payload['input_assessment'],
                                 {'category': 'off_topic', 'decision': 'block'})

    async def test_failures_are_excluded_ties_and_all_failures_are_false(self):
        cases = [
            (['{"safe":true}', '{"safe":true}', '{"safe":false}'], True, 3, 0),
            (['{"safe":false}', '{"safe":false}', '{"safe":true}'], False, 3, 0),
            (['{"safe":true}', '{"safe":false}', TimeoutError()], False, 2, 1),
            (['{"safe":true}', '{"safe":"true"}', RuntimeError()], True, 1, 2),
            ([TimeoutError(), '{"safe":1}', '{}'], False, 0, 3),
        ]
        for replies, expected, valid, errors in cases:
            p, state = self.case()
            state.errors = ['earlier:error']
            p.llm.chat.side_effect = replies
            await p._fusion_safety(state)
            self.assertIs(state.safe, expected)
            self.assertEqual(state.debug['fusion_completed_safety_votes'], valid)
            self.assertEqual(len(state.errors), errors + 1)
            self.assertEqual(state.errors[0], 'earlier:error')
            self.assertEqual(p.llm.chat.await_count, 3)

    async def test_stage_deadline_preserves_completed_votes_and_drains_hang(self):
        p, state = self.case()
        count, drained = 0, asyncio.Event()
        async def chat(*args, **kwargs):
            nonlocal count
            count += 1
            if count < 3:
                return '{"safe":true}'
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p.llm.chat.side_effect = chat
        result = await p._fusion_stage(p._fusion_safety(state), .02, state, 'safety')
        self.assertFalse(result)
        self.assertTrue(drained.is_set())
        self.assertTrue(state.safe)
        self.assertEqual(state.debug['fusion_completed_safety_votes'], 2)
        self.assertEqual(state.errors, ['safety:Deadline'])
        self.assertEqual(count, 3)

    async def test_external_cancel_drains_pending_votes_and_keeps_valid_result(self):
        p, state = self.case()
        started, drained = set(), set()
        ready = asyncio.Event()
        async def chat(*args, **kwargs):
            index = len(started)
            started.add(index)
            if index == 0:
                return '{"safe":true}'
            if len(started) == 3:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.add(index)
        p.llm.chat.side_effect = chat
        task = asyncio.create_task(p._fusion_safety(state))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(drained, {1, 2})
        self.assertTrue(state.safe)
        self.assertEqual(state.debug['fusion_completed_safety_votes'], 1)
        self.assertEqual(state.errors, [])
        self.assertEqual(p.llm.chat.await_count, 3)

    async def test_trial_mutations_and_unexpected_failures_are_isolated(self):
        p, state = self.case()
        state.unsupported = ['original span']
        state.citations = ['original source']
        before = deepcopy(state.__dict__)
        trials = []
        async def safety(instance, trial):
            index = len(trials)
            trials.append(trial)
            self.assertTrue(trial._material_budget_guard)
            self.assertTrue(trial.debug['fusion_input_reliable'])
            trial.label = Label.NORMAL
            trial.unsupported.append('trial span')
            trial.citations.append('trial source')
            trial.debug['trial'] = index
            trial.safe = True
            if index == 0:
                raise ValueError('unexpected parent failure')
        with patch.object(Fusion, '_fusion_safety', new=safety):
            await p._fusion_safety(state)
        self.assertTrue(state.safe)
        self.assertEqual(state.debug['fusion_completed_safety_votes'], 2)
        self.assertEqual(state.errors, ['fusion_safety_vote:ValueError'])
        for field in ('messages', 'candidate_answer', 'label', 'unsupported', 'citations'):
            self.assertEqual(getattr(state, field), before[field])
        self.assertNotIn('trial', state.debug)
        for field in ('errors', 'debug', 'unsupported', 'citations'):
            self.assertEqual(len({id(getattr(s, field)) for s in [state] + trials}), 4)

    async def test_nonboolean_without_error_cannot_enter_vote(self):
        p, state = self.case()
        values = iter([True, False, 'true'])
        async def safety(instance, trial):
            trial.safe = next(values)
        with patch.object(Fusion, '_fusion_safety', new=safety):
            await p._fusion_safety(state)
        self.assertFalse(state.safe)
        self.assertEqual(state.debug['fusion_completed_safety_votes'], 2)

    async def test_conditioned_unreliable_prior_stays_null_on_all_calls(self):
        p, state = self.case(VotedConditioned)
        state.debug['fusion_input_reliable'] = False
        p.llm.chat.return_value = '{"safe":false}'
        await p._fusion_safety(state)
        self.assertEqual(p.llm.chat.await_count, 3)
        for call in p.llm.chat.await_args_list:
            self.assertIsNone(json.loads(call.args[1])['input_assessment'])
            self.assertTrue(all(hint not in call.args[0] for hint in POLICY_HINTS.values()))

    def test_only_safety_votes_change_without_extra_budget_or_tutor_change(self):
        for cls, parent in ((VotedSafety, Fusion), (VotedConditioned, ConditionedSafety)):
            self.assertEqual(cls.guard_safety_votes, 3)
            self.assertEqual(cls.guard_safety_timeout, 13)
            self.assertEqual(cls.guard_deadline, 28)
            for method in ('tutor', '_run_safety', 'guard', '_fusion_ground', '_classify', '_generate',
                           '_fusion_safety_system', '_fusion_safety_payload'):
                self.assertIs(getattr(cls, method), getattr(parent, method))
            self.assertEqual(cls.guard_safety_after_input, parent.guard_safety_after_input)
        self.assertNotIn('guard_safety_votes', Fusion.__dict__)
        self.assertNotIn('guard_safety_votes', ConditionedSafety.__dict__)


if __name__ == '__main__':
    unittest.main()
