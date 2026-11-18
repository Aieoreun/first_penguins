"""Offline scheduling contracts, including cancellation during the fast gate."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.peace_guard_prompt import PEACE_BLOCK_SYSTEM
from pipelines.kimjaehak.peace_guard_scheduling import (
    FastGate, FastParallelInput, FastRequestedParallel, ParallelInput,
)
from pipelines.kimjaehak.peace_span_variants import RoutedSpan80


class GuardSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def body(self, candidate='Ordinary candidate answer.'):
        body = {'messages': [{'role': 'user', 'content': 'example request'}]}
        if candidate is not None:
            body['candidate_answer'] = candidate
        return body

    def pipeline(self, cls=FastGate):
        p = cls.__new__(cls)
        p._slots = asyncio.Semaphore(64)
        p.searcher = Searcher([Chunk('A', 'textbook evidence')])
        p.span_tagger = SimpleNamespace(predict=lambda text: ([], 0))
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"unsupported":[]}'))
        async def search(state, extra=''):
            state.chunks = p.searcher.chunks
        p._search = search
        return p

    async def test_immediate_block_skips_five_ground_calls_with_same_final_response(self):
        outputs = []
        for cls, calls in ((RoutedSpan80, 5), (FastGate, 0)):
            p = self.pipeline(cls)
            async def classify(state):
                state.label = Label.JAILBREAK_INJECTION
            p._classify = classify
            result = await p.guard(self.body())
            outputs.append(result.to_guard_response())
            self.assertEqual(p.llm.chat.await_count, calls)
            self.assertEqual(bool(result.debug.get('fusion_ground_skipped_fast_block')), calls == 0)
            self.assertTrue(result.grounded)
            self.assertEqual(result.unsupported, [])
            self.assertFalse(result.safe)
        self.assertEqual(outputs[0], outputs[1])

    async def test_actual_high_confidence_guard_classifier_still_runs_once(self):
        p = self.pipeline()
        predictions = []
        p.guard_vectorizer = SimpleNamespace(transform=lambda text: text)
        def predict(text):
            predictions.append(text)
            return np.array([[.99, .01]])
        p.guard_classifier = SimpleNamespace(classes_=np.array(['off_topic', 'normal']), predict_proba=predict)
        result = await p.guard(self.body())
        self.assertEqual(len(predictions), 1)
        self.assertTrue(result.debug['learned_guard_only'])
        self.assertTrue(result.debug['fusion_ground_skipped_fast_block'])
        self.assertEqual(result.label, Label.OFF_TOPIC)
        p.llm.chat.assert_not_awaited()

    async def test_skipped_ground_keeps_real_refusal_review_safety(self):
        p = self.pipeline()
        async def classify(state):
            state.label = Label.JAILBREAK_INJECTION
        p._classify = classify
        p.llm.chat.return_value = '{"safe":true}'
        result = await p.guard(self.body('도울 수는 없어요. 대신 안전한 학습을 안내할게요.'))
        self.assertTrue(result.debug['fusion_ground_skipped_fast_block'])
        self.assertTrue(result.safe)
        p.llm.chat.assert_awaited_once()
        self.assertEqual(p.llm.chat.await_args.kwargs['max_tokens'], 1800)

    async def test_allow_unknown_or_success_with_errors_does_not_skip(self):
        for label, error in ((Label.NORMAL, None), (None, None), (Label.OFF_TOPIC, 'material:TimeoutError')):
            p = self.pipeline()
            async def classify(state):
                state.label = label
                if error:
                    state.errors.append(error)
            p._classify = classify
            result = await p.guard(self.body())
            self.assertEqual(p.llm.chat.await_count, 5)
            self.assertFalse(result.debug.get('fusion_ground_skipped_fast_block', False))

    async def test_partial_block_then_exception_or_timeout_retains_unclassified_snapshot(self):
        for mode in ('exception', 'timeout'):
            p = self.pipeline()
            p.guard_fast_input_wait = .03
            p.guard_prepare_limit = .005
            snapshots = []
            original_ground = p._fusion_ground
            async def classify(state):
                state.label = Label.JAILBREAK_INJECTION
                if mode == 'exception':
                    raise RuntimeError('mock classification failure')
                await asyncio.Event().wait()
            async def ground(state):
                snapshots.append((state.label, state.section, state._material_budget_guard))
                await original_ground(state)
            p._classify, p._fusion_ground = classify, ground
            result = await p.guard(self.body())
            self.assertEqual(snapshots, [(None, 'A', True)])
            self.assertEqual(p.llm.chat.await_count, 5)
            self.assertFalse(result.debug.get('fusion_ground_skipped_fast_block', False))
            self.assertTrue(any(error.startswith('prepare:') for error in result.errors))

    async def test_unfinished_input_launches_ground_in_parallel_and_recomputes_remaining(self):
        p = self.pipeline()
        p.guard_fast_input_wait = .03
        p.guard_deadline = .2
        started = asyncio.Event()
        budgets = []
        original_stage = p._fusion_stage
        async def stage(coro, seconds, state, name):
            budgets.append((name, seconds))
            return await original_stage(coro, seconds, state, name)
        async def classify(state):
            await started.wait()
            state.label = Label.NORMAL
        async def ground(state):
            self.assertIsNone(state.label)
            started.set()
            state.grounded, state.unsupported = False, ['candidate']
        p._fusion_stage, p._classify, p._fusion_ground = stage, classify, ground
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertFalse(result.grounded)
        self.assertTrue(result.safe)
        self.assertFalse(result.debug.get('fusion_ground_skipped_fast_block', False))
        ground_budget = next(b for name, b in budgets if name == 'ground')
        self.assertGreater(ground_budget, 0)
        self.assertLess(ground_budget, .19)

    async def test_cancel_during_wait_drains_input_and_releases_slot(self):
        p = self.pipeline()
        p.guard_fast_input_wait = .5
        started, drained = asyncio.Event(), asyncio.Event()
        async def classify(state):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p._classify = classify
        p._fusion_ground = AsyncMock()
        task = asyncio.create_task(p.guard(self.body()))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(drained.is_set())
        self.assertEqual(p._slots._value, 64)
        p._fusion_ground.assert_not_awaited()

    async def test_outer_deadline_during_wait_drains_input(self):
        p = self.pipeline()
        p.guard_fast_input_wait = .5
        p.guard_deadline = .02
        drained = asyncio.Event()
        async def classify(state):
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p._classify = classify
        p._fusion_ground = AsyncMock()
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertTrue(drained.is_set())
        self.assertIn('request:Deadline', result.errors)
        self.assertFalse(result.safe)
        p._fusion_ground.assert_not_awaited()

    async def test_no_candidate_runs_only_original_input(self):
        p = self.pipeline()
        p._classify = AsyncMock()
        p._fusion_ground, p._fusion_safety = AsyncMock(), AsyncMock()
        result = await p.guard(self.body(None))
        p._classify.assert_awaited_once()
        p._fusion_ground.assert_not_awaited()
        p._fusion_safety.assert_not_awaited()
        self.assertNotIn('output', result.to_guard_response())

    async def test_concurrent_skip_and_pending_requests_keep_flags_isolated(self):
        p = self.pipeline()
        ground_started = asyncio.Event()
        seen = []
        async def classify(state):
            if state.last_user == 'fast block':
                state.label = Label.OFF_TOPIC
            else:
                await ground_started.wait()
                state.label = Label.NORMAL
        async def ground(state):
            seen.append(state.last_user)
            ground_started.set()
        p._classify, p._fusion_ground = classify, ground
        bodies = [self.body(), self.body()]
        bodies[0]['messages'][0]['content'] = 'fast block'
        bodies[1]['messages'][0]['content'] = 'pending allow'
        blocked, allowed = await asyncio.wait_for(asyncio.gather(*(p.guard(body) for body in bodies)), 1)
        self.assertEqual(seen, ['pending allow'])
        self.assertTrue(blocked.debug['fusion_ground_skipped_fast_block'])
        self.assertFalse(allowed.debug.get('fusion_ground_skipped_fast_block', False))
        self.assertFalse(blocked.safe)
        self.assertTrue(allowed.safe)
        self.assertEqual(p._slots._value, 64)

    async def test_parallel_input_material_starts_before_vote_group_completion(self):
        for cls in (ParallelInput, FastParallelInput, FastRequestedParallel):
            p = self.pipeline(cls)
            state = p._new_state(self.body(), True)
            material_started = asyncio.Event()
            async def vote(state, name, system, check):
                await material_started.wait()
                return [{'category': 'none'}]
            async def ask(state, name, system, payload, effort, timeout):
                self.assertEqual(name, 'material')
                material_started.set()
                return {'quote': 'evidence'}
            p._vote_group, p._ask = vote, ask
            result = await asyncio.wait_for(p._collect_checks(state, 'check', 'material'), 1)
            self.assertEqual(result[2], {'quote': 'evidence'})

    def test_thin_variant_mro_and_tutor_paths_are_unchanged(self):
        self.assertEqual(Fusion.guard_fast_input_wait, 0)
        self.assertEqual(RoutedSpan80.guard_fast_input_wait, 0)
        self.assertEqual(FastGate.guard_fast_input_wait, .005)
        for cls in (FastGate, ParallelInput, FastParallelInput, FastRequestedParallel):
            for name in ('tutor', '_classify', '_generate', '_fusion_ground', '_fusion_safety'):
                self.assertIs(getattr(cls, name), getattr(RoutedSpan80, name))
            self.assertEqual(cls.learned_span_threshold, .8)
            self.assertEqual(cls.learned_gate_threshold, RoutedSpan80.learned_gate_threshold)
        p = self.pipeline(FastRequestedParallel)
        guard, tutor = p._new_state(self.body(), True), p._new_state(self.body(), False)
        self.assertEqual(p._block_system(guard), PEACE_BLOCK_SYSTEM)
        self.assertEqual(p._block_system(tutor), RoutedSpan80._block_system(p, tutor))
        self.assertFalse(ParallelInput.guard_vote_first)
        self.assertFalse(FastParallelInput.guard_vote_first)
        self.assertFalse(FastRequestedParallel.guard_vote_first)


if __name__ == '__main__':
    unittest.main()
