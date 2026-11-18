"""Offline contracts for safety prompt/payload ablations, not model accuracy."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.peace_safety_variants import (
    ACTION_SAFETY_SYSTEM, ActionSafety, ContextSafety,
)


class PeaceSafetyVariantTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, cls, raw='{"action_summary":"설명", "unsafe_evidence":[], "safe":true}'):
        p = cls.__new__(cls)
        p._slots = asyncio.Semaphore(64)
        p.searcher = Searcher([Chunk('A', 'Lesson A\nfirst context'),
                               Chunk('B', 'Lesson B\nsecond context')])
        p.llm = SimpleNamespace(chat=AsyncMock(return_value=raw))
        state = p._new_state({'messages': [{'role': 'user', 'content': '예제의 동작을 설명해 줘'}],
                              'candidate_answer': '예제의 동작에 관한 설명입니다.'}, True)
        state.chunks = p.searcher.chunks
        state.section = 'B'  # Context is retrieved top one, not this field.
        return p, state

    async def test_real_safety_chat_serializes_each_payload_and_uses_same_limits(self):
        for cls in (ActionSafety, ContextSafety):
            p, state = self.pipeline(cls)
            before = deepcopy(state.__dict__)
            await p._fusion_safety(state)
            self.assertTrue(state.safe)
            p.llm.chat.assert_awaited_once()
            call = p.llm.chat.await_args
            self.assertEqual(call.args[0], ACTION_SAFETY_SYSTEM)
            payload = json.loads(call.args[1])
            expected = {'conversation': state.messages, 'candidate_answer': state.candidate_answer}
            if cls is ContextSafety:
                expected['curriculum_context'] = [{'section': 'A', 'text': p.searcher.section_text('A')}]
            self.assertEqual(payload, expected)
            self.assertEqual(call.kwargs, {'effort': 'low', 'max_tokens': 1800,
                                           'timeout': 13.0, 'json_mode': True})
            self.assertEqual(state.__dict__, before)

    async def test_extra_fields_do_not_override_strict_safe_boolean(self):
        for cls in (ActionSafety, ContextSafety):
            for safe, valid in [(True, True), (False, True), ('true', False), (1, False), (None, False)]:
                p, state = self.pipeline(cls, json.dumps({'action_summary': 'neutral',
                    'unsafe_evidence': [], 'safe': safe}))
                await p._fusion_safety(state)
                self.assertIs(state.safe, safe if valid else False)
                self.assertEqual(state.errors, [] if valid else ['independent_safety:ValueError'])
                self.assertEqual(p.llm.chat.await_count, 1)
            # No local rule turns a false model verdict into true based on evidence.
            p, state = self.pipeline(cls, '{"safe":false}')
            await p._fusion_safety(state)
            self.assertFalse(state.safe)
            self.assertEqual(state.errors, [])

    def test_context_payload_adds_only_first_section_without_mutation_or_search(self):
        p, state = self.pipeline(ContextSafety)
        original = Fusion._fusion_safety_payload(p, state)
        before = deepcopy(state.__dict__)
        p.searcher.search = Mock(side_effect=AssertionError('No new retrieval'))
        with patch.object(p.searcher, 'section_text', wraps=p.searcher.section_text) as read:
            payload = p._fusion_safety_payload(state)
            read.assert_called_once_with('A')
        self.assertEqual(set(payload) - set(original), {'curriculum_context'})
        self.assertNotIn('curriculum_context', original)
        self.assertEqual(state.__dict__, before)
        self.assertEqual(payload['conversation'], original['conversation'])
        state.chunks = []
        with patch.object(p.searcher, 'section_text', side_effect=AssertionError('No chunks')):
            self.assertEqual(p._fusion_safety_payload(state)['curriculum_context'], [])
        p.searcher.search.assert_not_called()

    async def test_two_requests_keep_safety_payload_and_parallel_output_states_isolated(self):
        for cls in (ActionSafety, ContextSafety):
            p, _ = self.pipeline(cls)
            copies, payloads, mutable_fields = [], [], []
            ready = asyncio.Event()
            def record(state):
                copies.append(state)
                # Keep references before the intentional final output merge.
                mutable_fields.append({field: getattr(state, field)
                                       for field in ('errors', 'debug', 'unsupported')})
                if len(copies) == 6:
                    ready.set()
            async def arrive(state):
                record(state)
                await ready.wait()
            async def search(state, extra=''):
                state.chunks = [p.searcher.chunks[0 if state.candidate_answer == 'first' else 1]]
            async def classify(state):
                await arrive(state)
                state.label = Label.NORMAL
            async def ground(state):
                await arrive(state)
                state.grounded = False
                state.unsupported = [state.candidate_answer]
            original_payload = cls._fusion_safety_payload
            def safety_payload(instance, state):
                # Capture the actual safety-owned state without replacing the call.
                record(state)
                return original_payload(instance, state)
            async def chat(system, user, **kwargs):
                await ready.wait()
                payload = json.loads(user)
                payloads.append(payload)
                return json.dumps({'action_summary': 'mock verdict', 'unsafe_evidence': [],
                                   'safe': payload['candidate_answer'] == 'first'})
            p._search, p._classify, p._fusion_ground = search, classify, ground
            p.llm.chat.side_effect = chat
            bodies = [{'messages': [{'role': 'user', 'content': name}], 'candidate_answer': name}
                      for name in ('first', 'second')]
            before = deepcopy(bodies)
            with patch.object(cls, '_fusion_safety_payload', new=safety_payload):
                results = await asyncio.wait_for(asyncio.gather(*(p.guard(body) for body in bodies)), 1)
            self.assertEqual([r.safe for r in results], [True, False])
            self.assertEqual([r.unsupported for r in results], [['first'], ['second']])
            self.assertTrue(all(r.label is Label.NORMAL and not r.grounded for r in results))
            self.assertEqual(p.llm.chat.await_count, 2)
            self.assertEqual(bodies, before)
            for field in ('errors', 'debug', 'unsupported'):
                self.assertEqual(len({id(row[field]) for row in mutable_fields}), 6)
            for payload in payloads:
                self.assertEqual(payload['conversation'][0]['content'], payload['candidate_answer'])
                if cls is ContextSafety:
                    self.assertEqual(payload['curriculum_context'][0]['section'],
                                     'A' if payload['candidate_answer'] == 'first' else 'B')
                else:
                    self.assertNotIn('curriculum_context', payload)

    async def test_timeout_and_cancel_retain_false_default_without_retry(self):
        for cls in (ActionSafety, ContextSafety):
            p, state = self.pipeline(cls)
            p.llm.chat.side_effect = TimeoutError()
            await p._fusion_safety(state)
            self.assertFalse(state.safe)
            self.assertEqual(state.errors, ['independent_safety:TimeoutError'])
            self.assertEqual(p.llm.chat.await_count, 1)
            p, state = self.pipeline(cls)
            p.llm.chat.side_effect = asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):
                await p._fusion_safety(state)
            self.assertFalse(state.safe)
            self.assertEqual(state.errors, [])
            self.assertEqual(p.llm.chat.await_count, 1)

    def test_other_methods_and_shared_fusion_prompt_remain_unchanged(self):
        self.assertNotEqual(ActionSafety.guard_safety_system, Fusion.guard_safety_system)
        self.assertEqual(ActionSafety.guard_safety_system, ContextSafety.guard_safety_system)
        for cls in (ActionSafety, ContextSafety):
            for name in ('guard', 'tutor', '_fusion_safety', '_fusion_ground', '_classify', '_generate',
                         '_collect_checks', '_ask', '_prepare_within'):
                self.assertIs(getattr(cls, name), getattr(Fusion, name))
            for name in ('guard_deadline', 'guard_prepare_limit', 'guard_safety_timeout',
                         'guard_ground_timeout', 'guard_material_timeout', 'output_votes'):
                self.assertEqual(getattr(cls, name), getattr(Fusion, name))


if __name__ == '__main__':
    unittest.main()
