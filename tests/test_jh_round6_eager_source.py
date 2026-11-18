import asyncio
from collections import Counter
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.v2 import prompts
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.round6_eager_source import Eager, Budget


class EagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_twelve_second_variant_changes_only_bounded_source_budget(self):
        for remaining, expected in ((26, 12), (2, 1.5)):
            p, s = self.setup_pipeline(cls=Budget)
            s._source_extract_deadline = time.monotonic() + remaining
            p._source_response = AsyncMock(return_value='{"section":"B"}')
            await p._classify(s)
            timeout = p._source_response.call_args.args[2]
            self.assertLessEqual(timeout, expected)
            self.assertGreater(timeout, expected - .1)
            await p._generate(s)
            p._source_response.assert_awaited_once()
            self.assertEqual(s.citations, ['B'])
            self.assertEqual(p.tutor_prepare_limit, 16)
            self.assertEqual(p.deadline, 26)
        self.assertIs(Budget._collect_checks, Eager._collect_checks)
        self.assertIs(Budget._ask, Base._ask)
        self.assertIs(Budget.guard, Base.guard)

    def setup_pipeline(self, cls=Eager, category='normal', confidence=.6, section_confidence=.8, guard=False):
        p = cls.__new__(cls)
        p.searcher = Searcher([Chunk('A', 'title A\nalpha'), Chunk('B', 'title B\nbeta')])
        p.vectorizer = p.section_vectorizer = SimpleNamespace(transform=lambda text: text)
        p.classifier = SimpleNamespace(classes_=np.array([category, 'out_of_material', 'off_topic']),
            predict_proba=lambda text: np.array([[confidence, (1-confidence)/2, (1-confidence)/2]]))
        p.section_classifier = SimpleNamespace(classes_=np.array(['A', 'B']),
            predict_proba=lambda text: np.array([[section_confidence, 1-section_confidence]]))
        p.guard_vectorizer, p.guard_classifier = p.vectorizer, p.classifier
        p._known_sections = {'A', 'B'}
        async def chat(system, user, **kwargs):
            if system == p.source_extract_system:
                return '{"section":"B"}'
            if system == prompts.MATERIAL_SYSTEM:
                return '{"quote":"alpha","section":"A"}'
            return '{"category":"none","self_info":false}'
        p.llm = SimpleNamespace(chat=AsyncMock(side_effect=chat))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'beta question'}]}, guard)
        s.chunks = p.searcher.chunks
        return p, s

    def call_counter(self, p):
        return Counter((c.args, json.dumps(c.kwargs, sort_keys=True)) for c in p.llm.chat.call_args_list)

    async def test_actual_mro_same_material_source_payload_and_final_response(self):
        base, original = self.setup_pipeline(cls=Base)
        await base._classify(original)
        await base._generate(original)
        p, s = self.setup_pipeline()
        await p._classify(s)
        self.assertTrue(s._eager_classification_complete)
        self.assertTrue(s._eager_source_reused)
        self.assertEqual(s.section, 'B')
        await p._generate(s)
        self.assertEqual(s.to_tutor_response(), original.to_tutor_response())
        self.assertEqual(self.call_counter(p), self.call_counter(base))
        self.assertEqual(p.llm.chat.await_count, 6)
        material = next(c for c in p.llm.chat.call_args_list if c.args[0] == prompts.MATERIAL_SYSTEM)
        self.assertEqual(material.args[1], prompts.material_user(s.chunks, s.request, s.history, s.last_user))
        self.assertIsNone(s._eager_source_task)

    async def test_source_overlaps_material_without_changing_material_call(self):
        p, s = self.setup_pipeline()
        source_started, material_started = asyncio.Event(), asyncio.Event()
        old = p.llm.chat.side_effect
        async def chat(system, user, **kwargs):
            if system == p.source_extract_system:
                source_started.set()
                await asyncio.wait_for(material_started.wait(), .5)
            elif system == prompts.MATERIAL_SYSTEM:
                material_started.set()
                await asyncio.wait_for(source_started.wait(), .5)
            return await old(system, user, **kwargs)
        p.llm.chat.side_effect = chat
        await p._classify(s)
        self.assertTrue(s._eager_source_reused)

    async def test_eligibility_and_category_gate_bypass(self):
        for category, probability, expected in [('normal', .49, False), ('normal', .5, True),
                ('pii_disclosure', .6, True), ('out_of_material', .6, False), ('normal', .96, False)]:
            p, s = self.setup_pipeline(category=category, confidence=probability)
            p._source_response = AsyncMock(return_value='{"section":"B"}')
            await p._classify(s)
            self.assertEqual(s._eager_source_attempted, expected)
            self.assertEqual(p._source_response.await_count, int(expected))
            if probability == .96:
                p.llm.chat.assert_not_awaited()
                self.assertTrue(s.debug['learned_gate'])

    async def test_unprefetched_paths_retain_original_source_selection(self):
        p, s = self.setup_pipeline(confidence=.49)
        await p._classify(s)
        self.assertFalse(s._eager_source_attempted)
        await p._generate(s)
        self.assertEqual(s.citations, ['B'])
        self.assertEqual(sum(c.args[0] == p.source_extract_system for c in p.llm.chat.call_args_list), 1)

    async def test_invalid_or_failed_source_uses_excerpt_without_second_call(self):
        for value in ('{}', '{"section":"missing"}', TimeoutError(), RuntimeError()):
            p, s = self.setup_pipeline()
            p._source_response = AsyncMock(**({'side_effect': value} if isinstance(value, Exception)
                                             else {'return_value': value}))
            await p._classify(s)
            self.assertFalse(s._eager_source_reused)
            self.assertEqual(s.section, 'A')
            await p._generate(s)
            self.assertEqual(s.citations, ['A'])
            p._source_response.assert_awaited_once()
            self.assertEqual(len(s.errors), 1)
            self.assertTrue(s.errors[0].startswith('source_extract:'))

    async def test_remaining_deadline_and_empty_pool_do_not_launch(self):
        p, s = self.setup_pipeline()
        s._source_extract_deadline = time.monotonic() + 2
        p._source_response = AsyncMock(return_value='{"section":"B"}')
        await p._classify(s)
        self.assertLessEqual(p._source_response.call_args.args[2], 1.5)
        self.assertEqual(p.deadline, 26)
        self.assertEqual(p.tutor_prepare_limit, 16)
        for ids, remaining in [([], 20), (['A'], 20), (['A', 'B'], .5)]:
            p, s = self.setup_pipeline()
            p._source_candidates = lambda view: ids
            p._source_response = AsyncMock()
            s._source_extract_deadline = time.monotonic() + remaining
            await p._classify(s)
            self.assertFalse(s._eager_source_attempted)
            p._source_response.assert_not_awaited()

    async def test_no_initial_section_does_not_prefetch_last_user_only_candidate(self):
        p, s = self.setup_pipeline()
        s.chunks = []
        p._source_response = AsyncMock()
        await p._classify(s)
        self.assertIsNone(s.section)
        self.assertFalse(s._eager_source_attempted)
        p._source_response.assert_not_awaited()

    async def test_attempted_source_preserves_no_section_and_long_excerpt_fallbacks(self):
        for section, excerpt in ((None, ''), ('B', 'x' * 4001)):
            p, s = self.setup_pipeline()
            s._eager_source_attempted = True
            s.section, s.label = section, Label.NORMAL
            p._excerpt = lambda state: excerpt
            p._source_response = AsyncMock()
            with patch.object(ContextFast, '_generate', new=AsyncMock()) as inherited:
                await p._generate(s)
                inherited.assert_awaited_once_with(s)
            self.assertEqual(s.section, section)
            p._source_response.assert_not_awaited()

    async def test_unchanged_material_false_and_section_gates_discard_prefetch(self):
        for outcome, confidence in [('unsupported', .8), ('normal', .91), ('normal', .99)]:
            p, s = self.setup_pipeline(section_confidence=confidence)
            old = p.llm.chat.side_effect
            async def chat(system, user, **kwargs):
                if outcome == 'unsupported' and system == prompts.MATERIAL_SYSTEM:
                    return '{"quote":"","section":"A"}'
                return await old(system, user, **kwargs)
            p.llm.chat.side_effect = chat
            await p._classify(s)
            self.assertTrue(s._eager_source_attempted)
            self.assertFalse(s._eager_source_reused)
            self.assertEqual(s.section, 'A')
            self.assertIsNone(s._eager_source_task)
            self.assertEqual(s.label, Label.OUT_OF_MATERIAL if outcome == 'unsupported' else Label.NORMAL)
            before = p.llm.chat.await_count
            await p._generate(s)
            self.assertEqual(p.llm.chat.await_count, before)

    async def test_classification_cancellation_drains_and_never_retries_source(self):
        p, s = self.setup_pipeline()
        started, drained = asyncio.Event(), asyncio.Event()
        async def source(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p._source_response = AsyncMock(side_effect=source)
        async def checks(*args):
            await started.wait()
            raise asyncio.CancelledError
        with patch.object(Base, '_collect_checks', new=checks):
            with self.assertRaises(asyncio.CancelledError):
                await p._classify(s)
        self.assertTrue(drained.is_set())
        self.assertFalse(s._eager_classification_complete)
        self.assertFalse(s._eager_source_reused)
        self.assertIsNone(s._eager_source_task)
        self.assertIsNone(s.section)  # Preserve the interrupted parent's state.
        with patch('pipelines.kimjaehak.round6_eager_source.FocusedExcerpt._generate',
                   new=AsyncMock()) as fallback:
            await p._generate(s)
            fallback.assert_awaited_once_with(s)
        p._source_response.assert_awaited_once()
        self.assertEqual(s.errors, [])

    async def test_actual_prepare_deadline_drains_and_keeps_parent_excerpt_fallback(self):
        p, s = self.setup_pipeline()
        p.tutor_prepare_limit = .02
        p._search = AsyncMock()
        started, drained = asyncio.Event(), asyncio.Event()
        async def source(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        p._source_response = AsyncMock(side_effect=source)
        old = p.llm.chat.side_effect
        async def chat(system, user, **kwargs):
            if system == prompts.MATERIAL_SYSTEM:
                await started.wait()
                await asyncio.Event().wait()
            return await old(system, user, **kwargs)
        p.llm.chat.side_effect = chat
        await p._prepare_within(s)
        self.assertTrue(drained.is_set())
        self.assertFalse(s._eager_classification_complete)
        self.assertIsNone(s._eager_source_task)
        self.assertEqual(s.section, 'A')  # Set by the unchanged prepare fallback.
        self.assertEqual(s.errors, ['prepare:Deadline'])
        await p._generate(s)
        self.assertEqual(s.citations, ['A'])
        p._source_response.assert_awaited_once()

    async def test_real_run_prepare_and_outer_deadlines_leave_no_source_task(self):
        for deadline, prepare, error in ((26, .02, 'prepare:Deadline'),
                                         (1.6, 2, 'request:Deadline')):
            p, _ = self.setup_pipeline()
            p.deadline, p.tutor_prepare_limit = deadline, prepare
            p._slots = asyncio.Semaphore(1)
            async def search(state, extra=''):
                state.chunks = p.searcher.chunks
            p._search = search
            started, drained = asyncio.Event(), asyncio.Event()
            async def source(*args):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()
            p._source_response = AsyncMock(side_effect=source)
            old = p.llm.chat.side_effect
            async def chat(system, user, **kwargs):
                if system == prompts.MATERIAL_SYSTEM:
                    await started.wait()
                    await asyncio.Event().wait()
                return await old(system, user, **kwargs)
            p.llm.chat.side_effect = chat
            state = await p.tutor({'messages': [{'role': 'user', 'content': 'beta question'}]})
            self.assertTrue(started.is_set() and drained.is_set())
            self.assertIsNone(state._eager_source_task)
            self.assertFalse(state._eager_classification_complete)
            self.assertFalse(state._eager_source_reused)
            self.assertIn(error, state.errors)
            p._source_response.assert_awaited_once()
            self.assertEqual(state.citations, ['A'] if error == 'prepare:Deadline' else [])

    async def test_classification_failure_and_nonnormal_result_drain_live_task(self):
        for failure in (True, False):
            p, s = self.setup_pipeline()
            started, drained = asyncio.Event(), asyncio.Event()
            async def source(*args):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()
            p._source_response = AsyncMock(side_effect=source)
            async def checks(*args):
                await started.wait()
                if failure:
                    raise RuntimeError('classification failed')
                return ([{'category': 'harmful_request'}] * 3,
                        [{'category': 'none', 'self_info': False}], {'quote': 'alpha'})
            with patch.object(Base, '_collect_checks', new=checks):
                if failure:
                    with self.assertRaises(RuntimeError):
                        await p._classify(s)
                else:
                    await p._classify(s)
                    self.assertEqual(s.label, Label.HARMFUL_REQUEST)
            self.assertTrue(drained.is_set())
            self.assertIsNone(s._eager_source_task)
            self.assertFalse(s._eager_source_reused)
            self.assertEqual(s.errors, [])

    async def test_optional_setup_errors_preserve_checks_and_do_not_mark_attempted(self):
        for stage in ('classifier', 'candidates', 'deadline'):
            p, s = self.setup_pipeline()
            if stage == 'classifier':
                p.classifier.predict_proba = lambda text: (_ for _ in ()).throw(ValueError('bad model'))
            elif stage == 'candidates':
                p._source_candidates = lambda view: (_ for _ in ()).throw(ValueError('bad candidates'))
            else:
                s._source_extract_deadline = 'bad deadline'
            expected = ([], [], {'quote': 'alpha'})
            with patch.object(Base, '_collect_checks', new=AsyncMock(return_value=expected)) as original:
                self.assertEqual(await p._collect_checks(s, 'check', 'original material'), expected)
                original.assert_awaited_once_with(s, 'check', 'original material')
            self.assertFalse(getattr(s, '_eager_source_attempted', False))
            self.assertEqual(len(s.errors), 1)
            self.assertTrue(s.errors[0].startswith('source_prefetch_setup:'))

    async def test_completed_errors_retained_for_nonnormal_gated_and_cancelled_checks(self):
        for outcome in ('normal', 'blocked', 'gated', 'cancelled'):
            p, s = self.setup_pipeline(section_confidence=.91 if outcome == 'gated' else .8)
            p._source_response = AsyncMock(side_effect=ValueError('source failure'))
            async def checks(view, state, *args):
                await state._eager_source_task
                if outcome == 'cancelled':
                    raise asyncio.CancelledError
                category = 'harmful_request' if outcome == 'blocked' else 'none'
                return ([{'category': category}] * 3, [{'category': 'none', 'self_info': False}], {'quote': 'alpha'})
            with patch.object(Base, '_collect_checks', new=checks):
                if outcome == 'cancelled':
                    with self.assertRaises(asyncio.CancelledError):
                        await p._classify(s)
                else:
                    await p._classify(s)
            self.assertEqual(s.errors, ['source_extract:ValueError'])
            self.assertIsNone(s._eager_source_task)

    async def test_concurrent_requests_do_not_share_selection_or_flags(self):
        p, first = self.setup_pipeline()
        second = p._new_state({'messages': [{'role': 'user', 'content': 'alpha question'}]}, False)
        second.chunks = first.chunks
        ready = set()
        both = asyncio.Event()
        async def source(view, ids, timeout):
            ready.add(view.last_user)
            if len(ready) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), .5)
            return json.dumps({'section': 'B' if 'beta' in view.last_user else 'A'})
        p._source_response = AsyncMock(side_effect=source)
        await asyncio.gather(p._classify(first), p._classify(second))
        self.assertEqual((first.section, second.section), ('B', 'A'))
        self.assertTrue(first._eager_source_reused and second._eager_source_reused)
        self.assertIsNone(first._eager_source_task)
        self.assertIsNone(second._eager_source_task)
        self.assertEqual(p._source_response.await_count, 2)

    async def test_guard_checks_and_generation_routing_unchanged(self):
        p, s = self.setup_pipeline(guard=True)
        p._source_response = AsyncMock()
        await p._classify(s)
        p._source_response.assert_not_awaited()
        self.assertFalse(getattr(s, '_eager_source_attempted', False))
        self.assertEqual(p.llm.chat.await_count, 5)
        with patch.object(Base, '_generate', new=AsyncMock()) as original:
            await p._generate(s)
            original.assert_awaited_once_with(s)
        self.assertIs(Eager.guard, Base.guard)
        self.assertIs(Eager._check_output, Base._check_output)


if __name__ == '__main__':
    unittest.main()
