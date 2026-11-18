"""Offline lifecycle and behavior contracts for the isolated guard fusion."""
import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.peace_guard_fusion import Pipeline, SAFETY_SYSTEM


class PeaceGuardFusionTests(unittest.IsolatedAsyncioTestCase):
    def body(self, candidate='wrong fact and more'):
        body = {'messages': [{'role': 'user', 'content': 'explain the lesson'}]}
        if candidate is not None:
            body['candidate_answer'] = candidate
        return body

    def pipeline(self):
        p = Pipeline.__new__(Pipeline)
        p._slots = asyncio.Semaphore(64)
        p.searcher = Searcher([Chunk('A', 'textbook evidence'), Chunk('B', 'other evidence')])
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"unsupported":["wrong fact"]}'))
        p.span_tagger = SimpleNamespace(predict=lambda candidate: ([], 0.0))
        return p

    def state(self, p):
        s = p._new_state(self.body(), True)
        s.chunks = p.searcher.chunks
        s.section = 'A'
        return s

    async def search(self, p, s, extra=''):
        self.assertEqual(extra, s.candidate_answer or '')
        s.chunks = p.searcher.chunks

    def test_parent_paths_and_budget_configuration_are_separate(self):
        for name in ('_classify', '_check_output', '_generate', 'tutor'):
            self.assertIs(getattr(Pipeline, name), getattr(Base, name))
        self.assertEqual(Pipeline.deadline, 26)
        self.assertEqual(Pipeline.guard_deadline, 28)
        self.assertEqual(Pipeline.guard_prepare_limit, 26)
        self.assertEqual(Pipeline.request_slots, 64)
        self.assertEqual(Pipeline.output_votes, 5)
        self.assertFalse(Pipeline.learned_section_use_for_output)
        for name in ('guard_ground_timeout', 'guard_safety_timeout', 'guard_material_timeout'):
            self.assertEqual(getattr(Pipeline, name), 13)

    async def test_ground_consensus_then_one_span_refinement(self):
        p = self.pipeline()
        s = self.state(p)
        s.safe = False
        predictions = []
        def predict(candidate):
            predictions.append(candidate)
            return ['wrong fact and more'], .99
        p.span_tagger.predict = predict
        await p._fusion_ground(s)
        self.assertEqual(p.llm.chat.await_count, 5)
        self.assertEqual(predictions, [s.candidate_answer])
        self.assertEqual(s.unsupported, ['wrong fact and more'])
        self.assertFalse(s.grounded)
        self.assertFalse(s.safe)
        self.assertTrue(s.debug.get('learned_span'))
        for call in p.llm.chat.call_args_list:
            self.assertEqual(call.args[0], p.unsupported_system)
            payload = json.loads(call.args[1])
            self.assertEqual([x['section'] for x in payload['교재']], ['A'])
            self.assertEqual(call.kwargs['max_tokens'], 1500)

    async def test_ground_provider_failure_keeps_successful_consensus(self):
        p = self.pipeline()
        s = self.state(p)
        calls = 0
        async def chat(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError('mock failure')
            return '{"unsupported":["wrong fact"]}'
        p.llm.chat.side_effect = chat
        await p._fusion_ground(s)
        self.assertEqual(calls, 5)
        self.assertEqual(s.unsupported, ['wrong fact'])
        self.assertFalse(s.grounded)
        self.assertTrue(any('RuntimeError' in e for e in s.errors))

    async def test_ground_cancel_preserves_four_completed_and_drains_fifth(self):
        p = self.pipeline()
        s = self.state(p)
        ready, drained = asyncio.Event(), asyncio.Event()
        calls = 0
        async def chat(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 5:
                return '{"unsupported":["wrong fact"]}'
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p.llm.chat.side_effect = chat
        task = asyncio.create_task(p._fusion_ground(s))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(drained.is_set())
        self.assertEqual(s.unsupported, ['wrong fact'])
        self.assertFalse(s.grounded)

    async def test_safety_is_one_independent_strict_boolean_call(self):
        for raw, expected in (('{"safe":true}', True), ('{"safe":false}', False),
                              ('{"safe":"true"}', False), ('{}', False)):
            p = self.pipeline()
            s = self.state(p)
            p.llm.chat.return_value = raw
            await p._fusion_safety(s)
            self.assertIs(s.safe, expected)
            p.llm.chat.assert_awaited_once()
            call = p.llm.chat.call_args
            self.assertEqual(call.args[0], SAFETY_SYSTEM)
            self.assertEqual(json.loads(call.args[1]), {
                'conversation': s.messages, 'candidate_answer': s.candidate_answer})
            self.assertEqual(call.kwargs['effort'], 'low')
            self.assertEqual(call.kwargs['max_tokens'], 1800)
            self.assertLessEqual(call.kwargs['timeout'], 13)
            self.assertEqual(bool(s.errors), raw in ('{"safe":"true"}', '{}'))
        p = self.pipeline()
        s = self.state(p)
        p.llm.chat.side_effect = RuntimeError('mock failure')
        await p._fusion_safety(s)
        self.assertFalse(s.safe)
        self.assertTrue(any('RuntimeError' in e for e in s.errors))
        p.llm.chat.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await p._fusion_safety(s)

    async def test_parallel_copies_flags_and_merge_field_ownership(self):
        for label in (Label.NORMAL, Label.JAILBREAK_INJECTION):
            p = self.pipeline()
            seen = []
            mutable_ids = []
            started = asyncio.Event()
            async def arrive(s):
                seen.append(s)
                mutable_ids.append({field: id(getattr(s, field))
                                    for field in ('errors', 'debug', 'unsupported')})
                if len(seen) == 3:
                    started.set()
                await started.wait()
            async def classify(s):
                await arrive(s)
                s.label = label
                s.debug['input'] = True
                s.errors.append('input:mock')
            async def ground(s):
                self.assertTrue(s._material_budget_guard)
                self.assertTrue(s._section_gate_guard)
                self.assertTrue(hasattr(s, '_source_extract_deadline'))
                await arrive(s)
                s.grounded, s.unsupported, s.safe = False, ['wrong fact'], True
                s.errors.append('ground:mock')
            async def safety(s):
                await arrive(s)
                s.safe = False
                s.label = Label.OFF_TOPIC
                s.errors.append('safety:mock')
            p._search = lambda s, extra='': self.search(p, s, extra)
            p._classify, p._fusion_ground, p._fusion_safety = classify, ground, safety
            result = await asyncio.wait_for(p.guard(self.body()), 1)
            self.assertEqual(result.label, label)
            self.assertFalse(result.safe)
            blocked = label is Label.JAILBREAK_INJECTION
            self.assertEqual(result.grounded, blocked)
            self.assertEqual(result.unsupported, [] if blocked else ['wrong fact'])
            self.assertCountEqual(result.errors, ['input:mock', 'ground:mock', 'safety:mock'])
            for field in ('errors', 'debug', 'unsupported'):
                self.assertEqual(len({row[field] for row in mutable_ids}), 3)
            p.llm.chat.assert_not_awaited()  # No extra original refusal review.

    async def test_real_guard_input_high_confidence_gate_is_retained(self):
        p = self.pipeline()
        p.guard_vectorizer = SimpleNamespace(transform=lambda text: text)
        p.guard_classifier = SimpleNamespace(classes_=np.array(['jailbreak_injection', 'normal']),
            predict_proba=lambda text: np.array([[.99, .01]]))
        p._search = lambda s, extra='': self.search(p, s, extra)
        async def ground(s):
            s.unsupported, s.grounded = ['wrong fact'], False
        async def safety(s):
            s.safe = True
        p._fusion_ground, p._fusion_safety = ground, safety
        result = await p.guard(self.body())
        self.assertTrue(result.debug['learned_guard_only'])
        self.assertEqual(result.label, Label.JAILBREAK_INJECTION)
        self.assertTrue(result.safe)
        self.assertTrue(result.grounded)
        self.assertEqual(result.unsupported, [])
        p.llm.chat.assert_not_awaited()

    async def test_external_cancel_drains_all_parallel_children(self):
        p = self.pipeline()
        p._search = lambda s, extra='': self.search(p, s, extra)
        begun, drained, ready = set(), set(), asyncio.Event()
        async def wait(name, s):
            begun.add(name)
            if len(begun) == 3:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.add(name)
        p._classify = lambda s: wait('input', s)
        p._fusion_ground = lambda s: wait('ground', s)
        p._fusion_safety = lambda s: wait('safety', s)
        task = asyncio.create_task(p.guard(self.body()))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(drained, begun)
        self.assertEqual(p._slots._value, 64)

    async def test_queue_wait_counts_toward_outer_deadline(self):
        p = self.pipeline()
        p._slots = asyncio.Semaphore(0)
        p.guard_deadline = .02
        p._search = AsyncMock()
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        p._search.assert_not_awaited()
        self.assertFalse(result.safe)
        self.assertTrue(any('Deadline' in e for e in result.errors))

    async def test_guard_ground_timeout_keeps_completed_votes_and_safety(self):
        p = self.pipeline()
        p.guard_ground_timeout = .02
        p._search = lambda s, extra='': self.search(p, s, extra)
        async def classify(s):
            s.label = Label.NORMAL
        p._classify = classify
        calls, drained = 0, asyncio.Event()
        async def chat(system, *args, **kwargs):
            nonlocal calls
            if system == SAFETY_SYSTEM:
                return '{"safe":false}'
            calls += 1
            if calls < 5:
                return '{"unsupported":["wrong fact"]}'
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p.llm.chat.side_effect = chat
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertTrue(drained.is_set())
        self.assertEqual(result.unsupported, ['wrong fact'])
        self.assertFalse(result.grounded)
        self.assertFalse(result.safe)
        self.assertEqual(result.debug['fusion_completed_ground_votes'], 4)
        self.assertIn('ground:Deadline', result.errors)
        self.assertEqual(p.llm.chat.await_count, 6)

    async def test_no_candidate_only_runs_input(self):
        p = self.pipeline()
        p._search = lambda s, extra='': self.search(p, s, extra)
        p._classify = AsyncMock()
        p._fusion_ground, p._fusion_safety = AsyncMock(), AsyncMock()
        result = await p.guard(self.body(None))
        p._classify.assert_awaited_once()
        p._fusion_ground.assert_not_awaited()
        p._fusion_safety.assert_not_awaited()
        self.assertNotIn('output', result.to_guard_response())

    async def test_conditioned_safety_waits_for_input_but_ground_is_parallel(self):
        p = self.pipeline()
        p.guard_safety_after_input = True
        p._search = lambda s, extra='': self.search(p, s, extra)
        ground_started = asyncio.Event()
        async def ground(s):
            ground_started.set()
        async def classify(s):
            await ground_started.wait()
            s.label = Label.OFF_TOPIC
        async def safety(s):
            self.assertTrue(ground_started.is_set())
            self.assertEqual(s.label, Label.OFF_TOPIC)
            self.assertTrue(s.debug['fusion_input_reliable'])
            s.safe = True
        p._classify, p._fusion_ground, p._fusion_safety = classify, ground, safety
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertTrue(result.safe)
        self.assertEqual(result.label, Label.OFF_TOPIC)

    async def test_failed_classification_is_not_a_reliable_normal_prior(self):
        p = self.pipeline()
        p.guard_safety_after_input = True
        p._search = lambda s, extra='': self.search(p, s, extra)
        p._classify = AsyncMock(side_effect=ValueError('fixture'))
        p._fusion_ground = AsyncMock()
        async def safety(s):
            self.assertFalse(s.debug['fusion_input_reliable'])
            s.safe = True
        p._fusion_safety = safety
        result = await p.guard(self.body())
        self.assertIn('prepare:ValueError', result.errors)
        self.assertTrue(result.safe)

    async def test_conditioned_safety_never_starts_with_expired_request_budget(self):
        p = self.pipeline()
        p.guard_safety_after_input = True
        p._fusion_safety = AsyncMock()
        source, safety = self.state(p), self.state(p)
        completed = asyncio.create_task(asyncio.sleep(0, result=True))
        await p._run_safety(safety, source, completed, time.monotonic() - 29)
        p._fusion_safety.assert_not_awaited()
        self.assertIn('safety:NoBudget', safety.errors)

    async def test_conditioned_cancel_drains_input_and_ground_before_safety(self):
        p = self.pipeline()
        p.guard_safety_after_input = True
        p._search = lambda s, extra='': self.search(p, s, extra)
        begun, drained = set(), set()
        ready = asyncio.Event()
        async def waiting(name, state):
            begun.add(name)
            if len(begun) == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.add(name)
        p._classify = lambda s: waiting('input', s)
        p._fusion_ground = lambda s: waiting('ground', s)
        p._fusion_safety = AsyncMock()
        task = asyncio.create_task(p.guard(self.body()))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(drained, {'input', 'ground'})
        p._fusion_safety.assert_not_awaited()
        self.assertEqual(p._slots._value, 64)

    async def test_guard_votes_finish_before_material_and_tutor_delegates(self):
        p = self.pipeline()
        s = self.state(p)
        completed = set()
        async def vote(state, name, system, payload):
            await asyncio.sleep(0)
            completed.add(name)
            return [{'category': 'none'}]
        async def ask(state, name, system, payload, effort, timeout):
            self.assertEqual(completed, {'block', 'other'})
            self.assertEqual((name, payload, timeout), ('material', 'material payload', 13))
            return {'quote': 'evidence'}
        p._vote_group, p._ask = vote, ask
        result = await p._collect_checks(s, 'check payload', 'material payload')
        self.assertEqual(result[2], {'quote': 'evidence'})
        tutor = p._new_state(self.body(None), False)
        with patch.object(Base, '_collect_checks', new=AsyncMock(return_value='parent')) as original:
            self.assertEqual(await p._collect_checks(tutor, 'check', 'material'), 'parent')
            original.assert_awaited_once_with(tutor, 'check', 'material')


if __name__ == '__main__':
    unittest.main()
