"""Offline disagreement-review tests; the learned hint is never the verdict."""
import asyncio
from copy import deepcopy
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion, isolated_state
from pipelines.kimjaehak.peace_guard_scheduling import FastParallelInput
from pipelines.kimjaehak.peace_span_variants import CurrentSpan80
from pipelines.kimjaehak.peace_span_review import CurrentReview, FastReview


class SpanReviewTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self, cls=CurrentReview, review='{"unsupported":["wrong detail"]}'):
        p = cls.__new__(cls)
        p._slots = asyncio.Semaphore(64)
        p.searcher = Searcher([Chunk('A', 'The textbook provides the correct detail.')])
        p.span_tagger = SimpleNamespace(predict=lambda text: (['wrong detail'], .97))
        async def chat(system, *args, **kwargs):
            return review if system == p.guard_span_review_system else '{"unsupported":[]}'
        p.llm = SimpleNamespace(chat=AsyncMock(side_effect=chat))
        body = {'messages': [{'role': 'user', 'content': 'Explain this detail.'}],
                'candidate_answer': 'This is a wrong detail.'}
        s = p._new_state(body, True)
        s.label, s.chunks, s.section = Label.NORMAL, p.searcher.chunks, 'A'
        return p, s, body

    async def review(self, p, s, deadline=None):
        await p._review_span_disagreement(s, time.monotonic() + 20 if deadline is None else deadline)

    async def test_actual_current_mro_five_votes_then_one_review(self):
        p, s, _ = self.setup_pipeline()
        await p._check_output(s)
        self.assertEqual(p.llm.chat.await_count, 6)
        self.assertFalse(s.grounded)
        self.assertEqual(s.unsupported, ['wrong detail'])
        self.assertTrue(s.safe)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertTrue(s.debug['span_review_completed'])
        self.assertTrue(s.debug['span_review_success'])
        self.assertTrue(s.debug['span_review_changed'])
        call = p.llm.chat.await_args
        self.assertEqual(call.args[0], p.guard_span_review_system)
        self.assertTrue(p.guard_span_review_system.startswith(p.unsupported_system))
        original = p._ground_payload(s)
        self.assertEqual(json.loads(call.args[1]), {**original, '검토_대상_구절': ['wrong detail']})
        self.assertEqual(call.kwargs, {'effort': 'low', 'max_tokens': 1500, 'timeout': 6.0, 'json_mode': True})

    async def test_empty_verdict_overrules_hint_without_changing_original(self):
        p, s, _ = self.setup_pipeline(review='{"unsupported":[]}')
        s.safe = False
        await self.review(p, s)
        self.assertTrue(s.grounded)
        self.assertEqual(s.unsupported, [])
        self.assertFalse(s.safe)
        self.assertTrue(s.debug['span_review_completed'])
        p.llm.chat.assert_awaited_once()

    async def test_exact_llm_span_commits_without_second_refinement(self):
        p, s, _ = self.setup_pipeline(review='{"unsupported":["detail","detail"]}')
        calls = []
        def predict(text):
            calls.append(text)
            return ['wrong detail'], .99
        p.span_tagger.predict = predict
        s.safe = False
        before = deepcopy((s.messages, s.chunks, s.label, s.section))
        await self.review(p, s)
        self.assertEqual(s.unsupported, ['detail'])
        self.assertEqual(calls, [s.candidate_answer])
        self.assertFalse(s.safe)
        self.assertEqual((s.messages, s.chunks, s.label, s.section), before)

    async def test_skip_boundaries_and_ineligible_states_make_no_call(self):
        cases = ('low', 'no_prediction', 'invalid_prediction', 'nan', 'block', 'no_candidate',
                 'ungrounded', 'existing_span', 'tutor', 'budget', 'no_deadline')
        for case in cases:
            p, s, _ = self.setup_pipeline()
            deadline = time.monotonic() + 20
            if case == 'low': p.span_tagger.predict = lambda text: (['wrong detail'], .94999)
            elif case == 'no_prediction': p.span_tagger.predict = lambda text: ([], .99)
            elif case == 'invalid_prediction': p.span_tagger.predict = lambda text: (['absent'], .99)
            elif case == 'nan': p.span_tagger.predict = lambda text: (['wrong detail'], float('nan'))
            elif case == 'block': s.label = Label.JAILBREAK_INJECTION
            elif case == 'no_candidate': s.candidate_answer = None
            elif case == 'ungrounded': s.grounded = False
            elif case == 'existing_span': s.unsupported = ['detail']
            elif case == 'tutor': s._material_budget_guard = False
            elif case == 'budget': deadline = time.monotonic() + 1
            elif case == 'no_deadline': deadline = None
            old = deepcopy((s.grounded, s.unsupported, s.safe, s.label))
            await p._review_span_disagreement(s, deadline)
            p.llm.chat.assert_not_awaited()
            self.assertEqual((s.grounded, s.unsupported, s.safe, s.label), old)
        p, s, _ = self.setup_pipeline()
        p.span_tagger.predict = lambda text: (['wrong detail'], .95)
        await self.review(p, s)
        p.llm.chat.assert_awaited_once()

    async def test_invalid_review_is_atomic_and_does_not_retry(self):
        for raw in ('not json', '{}', '{"unsupported":"detail"}', '{"unsupported":[1]}',
                    '{"unsupported":[""]}', '{"unsupported":["wrong detail","not present"]}',
                    '{"unsupported":["WRONG DETAIL"]}'):
            p, s, _ = self.setup_pipeline(review=raw)
            s.safe = False
            await self.review(p, s)
            self.assertTrue(s.grounded)
            self.assertEqual(s.unsupported, [])
            self.assertFalse(s.safe)
            self.assertEqual(s.errors, ['span_review:ValueError'])
            self.assertTrue(s.debug['span_review_completed'])
            self.assertFalse(s.debug['span_review_success'])
            p.llm.chat.assert_awaited_once()

    async def test_provider_setup_failure_and_cancellation_preserve_verdict(self):
        for error in (RuntimeError('mock'), asyncio.CancelledError()):
            p, s, _ = self.setup_pipeline()
            p.llm.chat.side_effect = error
            if isinstance(error, asyncio.CancelledError):
                with self.assertRaises(asyncio.CancelledError):
                    await self.review(p, s)
                self.assertEqual(s.errors, [])
            else:
                await self.review(p, s)
                self.assertEqual(s.errors, ['span_review:RuntimeError'])
            self.assertTrue(s.grounded)
            self.assertEqual(s.unsupported, [])
        p, s, _ = self.setup_pipeline()
        def fail(text): raise RuntimeError('model failure')
        p.span_tagger.predict = fail
        await self.review(p, s)
        p.llm.chat.assert_not_awaited()
        self.assertTrue(s.grounded)
        self.assertEqual(s.errors, ['span_review_setup:RuntimeError'])

    async def test_inner_timeout_drains_and_remaining_budget_caps_call(self):
        p, s, _ = self.setup_pipeline()
        await self.review(p, s, time.monotonic() + 2)
        self.assertGreater(p.llm.chat.await_args.kwargs['timeout'], 1)
        self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'], 1.5)
        p, s, _ = self.setup_pipeline()
        p.guard_span_review_timeout, p.span_review_min_budget = .02, 0
        drained = asyncio.Event()
        async def hang(*args, **kwargs):
            try: await asyncio.Event().wait()
            finally: drained.set()
        p.llm.chat.side_effect = hang
        await asyncio.wait_for(self.review(p, s), 1)
        self.assertTrue(drained.is_set())
        self.assertTrue(s.grounded)
        self.assertEqual(s.unsupported, [])
        self.assertEqual(s.errors, ['span_review:TimeoutError'])
        self.assertFalse(s.debug['span_review_completed'])
        self.assertFalse(s.debug['span_review_success'])

    async def test_current_original_grounding_error_prevents_review(self):
        p, s, _ = self.setup_pipeline()
        original = p.llm.chat.side_effect
        calls = 0
        async def chat(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1: raise RuntimeError('one grounding failure')
            return await original(*args, **kwargs)
        p.llm.chat.side_effect = chat
        await p._check_output(s)
        self.assertEqual(calls, 5)
        self.assertTrue(s.grounded)
        self.assertNotIn('span_review_attempted', s.debug)

    async def test_fast_hook_requires_three_good_votes_and_final_nonblock(self):
        for count, label, expected in ((2, Label.NORMAL, 0), (3, Label.NORMAL, 1),
                                       (5, Label.JAILBREAK_INJECTION, 0)):
            p, s, _ = self.setup_pipeline(FastReview)
            s.label = label
            ground = isolated_state(s)
            ground.label = None  # Grounding snapshot precedes input classification.
            ground.debug['fusion_completed_ground_votes'] = count
            await p._fusion_post_checks(s, ground, time.monotonic())
            self.assertEqual(p.llm.chat.await_count, expected)
            self.assertEqual(ground.grounded, not bool(expected))
            self.assertEqual(s.label, label)

    async def test_actual_fast_guard_post_hook_merges_review_and_preserves_safety(self):
        p, _, body = self.setup_pipeline(FastReview)
        async def search(s, extra=''): s.chunks = p.searcher.chunks
        async def classify(s): s.label = Label.NORMAL
        p._search, p._classify = search, classify
        result = await p.guard(body)
        self.assertEqual(p.llm.chat.await_count, 6)
        self.assertFalse(result.grounded)
        self.assertEqual(result.unsupported, ['wrong detail'])
        self.assertEqual(result.label, Label.NORMAL)
        self.assertTrue(result.safe)
        self.assertTrue(result.debug['span_review_completed'])

    async def test_fast_external_cancel_during_post_hook_drains_review(self):
        p, _, body = self.setup_pipeline(FastReview)
        entered, drained = asyncio.Event(), asyncio.Event()
        async def search(s, extra=''): s.chunks = p.searcher.chunks
        async def classify(s): s.label = Label.NORMAL
        async def chat(system, *args, **kwargs):
            if system != p.guard_span_review_system: return '{"unsupported":[]}'
            entered.set()
            try: await asyncio.Event().wait()
            finally: drained.set()
        p._search, p._classify, p.llm.chat.side_effect = search, classify, chat
        task = asyncio.create_task(p.guard(body))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue(drained.is_set())
        self.assertEqual(p._slots._value, 64)

    async def test_parallel_requests_do_not_share_review_verdict(self):
        p, yes, _ = self.setup_pipeline()
        _, no, _ = self.setup_pipeline()
        no.candidate_answer = 'An ordinary statement.'
        p.span_tagger.predict = lambda text: (['wrong detail'], .97) if 'wrong' in text else ([], .99)
        await asyncio.gather(self.review(p, yes), self.review(p, no))
        self.assertFalse(yes.grounded)
        self.assertTrue(no.grounded)
        p.llm.chat.assert_awaited_once()

    async def test_actual_request_budget_bounds_review_for_both_schedulers(self):
        for cls in (CurrentReview, FastReview):
            p, _, body = self.setup_pipeline(cls)
            p.deadline = p.guard_deadline = .04
            p.span_review_deadline_margin = p.span_review_min_budget = 0
            entered, drained = asyncio.Event(), asyncio.Event()
            async def search(s, extra=''): s.chunks = p.searcher.chunks
            async def classify(s): s.label = Label.NORMAL
            async def prepare(s, extra=''):
                s.chunks, s.section, s.label = p.searcher.chunks, 'A', Label.NORMAL
            async def chat(system, *args, **kwargs):
                if system != p.guard_span_review_system: return '{"unsupported":[]}'
                entered.set()
                try: await asyncio.Event().wait()
                finally: drained.set()
            p._search, p._classify, p._prepare_within = search, classify, prepare
            p.llm.chat.side_effect = chat
            result = await asyncio.wait_for(p.guard(body), 1)
            self.assertTrue(entered.is_set())
            self.assertTrue(drained.is_set())
            self.assertTrue(result.grounded)
            self.assertEqual(result.unsupported, [])
            self.assertEqual(result.label, Label.NORMAL)
            self.assertTrue(result.safe)
            self.assertEqual(p.llm.chat.await_count, 6)
            self.assertTrue(result.errors)

    async def test_default_hook_and_tutor_paths_remain_unchanged(self):
        base, s, _ = self.setup_pipeline(FastReview)
        before = deepcopy(vars(s))
        await Fusion._fusion_post_checks(base, s, None, time.monotonic())
        self.assertEqual(vars(s), before)
        base.llm.chat.assert_not_awaited()
        for cls, parent in ((CurrentReview, CurrentSpan80), (FastReview, FastParallelInput)):
            for name in ('tutor', '_generate', '_classify'):
                self.assertIs(getattr(cls, name), getattr(parent, name))
            self.assertEqual(cls.learned_span_threshold, .8)
            self.assertEqual(cls.deadline, 26)
        self.assertEqual(FastReview.guard_deadline, 28)


if __name__ == '__main__':
    unittest.main()
