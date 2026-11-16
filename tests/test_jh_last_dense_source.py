import asyncio
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from core.schema import Label
from core.state import Chunk
from pipelines.kimjaehak.extractive_answer import Section
from pipelines.kimjaehak.focused_excerpt import Selected
from pipelines.kimjaehak.last_dense_source import Pipeline
from pipelines.kimjaehak.source_extract import SelectedSection
from pipelines.v1.pipeline import QUERY_MAX_CHARS


class LastDenseSourceTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self):
        p = Pipeline.__new__(Pipeline)
        hits = [Chunk(s, s + ' material') for s in ('dense1', 'dense1', 'dense2', 'dense3')]
        p.searcher = SimpleNamespace(has_embeddings=True,
            search=Mock(side_effect=lambda text, vector=None, **kw: hits if vector is not None else [Chunk('last', 'material')]),
            section_text=lambda sid: sid + ' material')
        p.llm = SimpleNamespace(can_embed=True, embed=AsyncMock(return_value=[[.1, .9]]),
                                chat=AsyncMock(return_value='{"section":"dense2"}'))
        # The focused excerpt renderer is unrelated to this candidate's pool.
        p._excerpt = lambda state: p.searcher.section_text(state.section)
        s = p._new_state({'messages': [{'role': 'user', 'content': 'original last question'}]}, False)
        s.label, s.section = Label.NORMAL, 'old'
        s.chunks = [Chunk(sid, sid) for sid in ('old', 'A', 'B', 'C', 'D')]
        return p, s

    async def test_default_async_hook_is_exact_old_sync_candidate_result(self):
        p, s = self.make_case()
        expected = p._source_candidates(s)
        p.searcher.search.reset_mock()
        self.assertEqual(await SelectedSection._candidate_sections(p, s), expected)
        p.searcher.search.assert_called_once_with(s.last_user, bm25_top=1, embedding_top=0)
        p.llm.embed.assert_not_awaited()

    async def test_dense_distinct_top_two_insert_after_base_prefix_and_cap_six(self):
        p, s = self.make_case()
        self.assertEqual(await p._candidate_sections(s), ['old', 'last', 'dense1', 'dense2', 'A', 'B'])
        p.llm.embed.assert_awaited_once_with([s.last_user], 3.0)
        self.assertEqual(p.searcher.search.call_args_list, [
            call(s.last_user, bm25_top=1, embedding_top=0),
            call(s.last_user, [.1, .9], bm25_top=0, embedding_top=10)])
        self.assertEqual(s.section, 'old')
        self.assertEqual([c.section_id for c in s.chunks], ['old', 'A', 'B', 'C', 'D'])

    async def test_duplicate_base_hits_consume_top_two_dense_slots_without_extra_candidates(self):
        p, s = self.make_case()
        p.searcher.search.side_effect = lambda text, vector=None, **kw: [Chunk(x, x) for x in
            (('last', 'last', 'C', 'dense3') if vector is not None else ('last',))]
        self.assertEqual(await p._candidate_sections(s), ['old', 'last', 'C', 'A', 'B', 'D'])

    async def test_embedding_query_is_truncated_but_search_and_messages_are_original(self):
        p, s = self.make_case()
        s.messages[-1]['content'] = '가' * (QUERY_MAX_CHARS + 30)
        original = s.last_user
        await p._candidate_sections(s)
        self.assertEqual(p.llm.embed.await_args.args[0], [original[:QUERY_MAX_CHARS]])
        self.assertEqual(p.searcher.search.call_args.args[0], original)
        self.assertEqual(s.last_user, original)

    async def test_missing_embeddings_low_budget_and_empty_vectors_restore_exact_pool(self):
        for case in ('index_off', 'provider_off', 'budget', 'no_deadline', 'none', 'empty', 'empty_vector'):
            with self.subTest(case=case):
                p, s = self.make_case()
                expected = p._source_candidates(s)
                if case == 'index_off': p.searcher.has_embeddings = False
                if case == 'provider_off': p.llm.can_embed = False
                if case == 'budget': s._source_extract_deadline = time.monotonic() + 3.9
                if case == 'no_deadline': del s._source_extract_deadline
                if case == 'none': p.llm.embed.return_value = None
                if case == 'empty': p.llm.embed.return_value = []
                if case == 'empty_vector': p.llm.embed.return_value = [[]]
                self.assertEqual(await p._candidate_sections(s), expected)
                if case in ('index_off', 'provider_off', 'budget', 'no_deadline'):
                    p.llm.embed.assert_not_awaited()

    async def test_embedding_or_dense_search_error_falls_back_once_and_preserves_errors(self):
        for stage in ('embed', 'search'):
            p, s = self.make_case()
            s.errors = ['existing:error']
            expected = p._source_candidates(s)
            if stage == 'embed': p.llm.embed.side_effect = TimeoutError()
            else:
                def search(text, vector=None, **kw):
                    if vector is not None: raise ValueError('private detail')
                    return [Chunk('last', 'material')]
                p.searcher.search.side_effect = search
            self.assertEqual(await p._candidate_sections(s), expected)
            self.assertEqual(s.errors, ['existing:error', 'source_last_dense:' + ('TimeoutError' if stage == 'embed' else 'ValueError')])
            p.llm.embed.assert_awaited_once()

    async def test_generation_one_selector_same_prompt_and_eight_second_cap(self):
        for label in (Label.NORMAL, Label.PII_DISCLOSURE):
            p, s = self.make_case()
            s.label = label
            await p._generate(s)
            p.llm.embed.assert_awaited_once()
            p.llm.chat.assert_awaited_once()
            args, kwargs = p.llm.chat.await_args
            self.assertEqual(args[0], Selected.source_extract_system)
            self.assertEqual(kwargs, dict(effort='low', max_tokens=2000, timeout=8, json_mode=True))
            payload = json.loads(args[1])
            self.assertEqual(payload['conversation'], s.messages)
            self.assertEqual([x['section'] for x in payload['materials']], ['old', 'last', 'dense1', 'dense2', 'A', 'B'])
            self.assertEqual(s.citations, ['dense2'])

    async def test_outer_embedding_timeout_bounds_a_provider_that_does_not_finish(self):
        p, s = self.make_case()
        expected = p._source_candidates(s)
        p.source_last_dense_timeout = .01
        cancelled = asyncio.Event()
        async def embed(*args):
            try: await asyncio.Event().wait()
            finally: cancelled.set()
        p.llm.embed.side_effect = embed
        self.assertEqual(await p._candidate_sections(s), expected)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(s.errors, ['source_last_dense:TimeoutError'])
        p.llm.embed.assert_awaited_once_with([s.last_user], .01)

    async def test_remaining_budget_is_recomputed_after_embedding(self):
        p, s = self.make_case()
        s._source_extract_deadline = 126.
        # Hook starts with 4s remaining; embedding consumes 3s. The inherited
        # selector sees only .5s after its reserve and must skip the chat call.
        with patch('pipelines.kimjaehak.last_dense_source.time.monotonic', side_effect=[122., 125., 125.]):
            p.llm.embed = AsyncMock(return_value=[[.1, .9]])
            # Avoid asyncio's own monotonic calls while retaining its await.
            async def immediate(coro, timeout): return await coro
            with patch('pipelines.kimjaehak.last_dense_source.asyncio.wait_for', side_effect=immediate):
                await p._generate(s)
        p.llm.embed.assert_awaited_once_with([s.last_user], 3.0)
        p.llm.chat.assert_not_awaited()
        self.assertEqual(s.citations, ['old'])

    async def test_guard_policy_and_section_gates_bypass_embedding_and_selector(self):
        cases = [(True, Label.NORMAL, {}, 'old'), (False, Label.NORMAL, {}, None),
                 (False, Label.NORMAL, {'learned_section': True}, 'old'),
                 (False, Label.NORMAL, {'tutor_section_gate': .90}, 'old')]
        cases += [(False, label, {}, 'old') for label in Label if label not in {Label.NORMAL, Label.PII_DISCLOSURE}]
        for guard, label, debug, section in cases:
            p, s = self.make_case()
            s._extract_guard, s.label, s.debug, s.section = guard, label, debug, section
            with patch.object(Section, '_generate', new_callable=AsyncMock):
                await p._generate(s)
            p.searcher.search.assert_not_called()
            p.llm.embed.assert_not_awaited()
            p.llm.chat.assert_not_awaited()

    async def test_cancellation_propagates_without_fallback_selector(self):
        p, s = self.make_case()
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def embed(*args):
            entered.set()
            try: await asyncio.Event().wait()
            finally: cancelled.set()
        p.llm.embed.side_effect = embed
        task = asyncio.create_task(p._generate(s))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue(cancelled.is_set())
        self.assertEqual(s.errors, [])
        p.llm.chat.assert_not_awaited()

    def test_twenty_six_second_deadline_and_other_paths_are_inherited(self):
        p, s = self.make_case()
        self.assertLessEqual(s._source_extract_deadline - time.monotonic(), 26)
        self.assertGreater(s._source_extract_deadline - time.monotonic(), 25)
        self.assertEqual(Pipeline.deadline, 26)
        self.assertEqual(Pipeline.source_extract_sections, 6)
        for name in ('guard', '_classify', '_search', '_run', '_prepare_within', '_check_output'):
            self.assertIs(getattr(Pipeline, name), getattr(Selected, name))


if __name__ == '__main__':
    unittest.main()
