import asyncio
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import FALLBACK_ANSWER, Label
from core.state import Chunk
from pipelines.kimjaehak.focused_excerpt import Selected
from pipelines.kimjaehak.material_budget import Pipeline


class MaterialBudgetTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self, guard=False):
        p = Pipeline.__new__(Pipeline)
        p.searcher = object()
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"quote":"source","section":"S"}'))
        p._slots = asyncio.Semaphore(2)
        s = p._new_state({'messages': [{'role': 'user', 'content': 'original question'}]}, guard)
        s.chunks = [Chunk('S', 'source')]
        return p, s

    async def test_tutor_material_real_parent_ask_has_same_prompt_tokens_and_one_call(self):
        p, s = self.make_case()
        before = list(s.messages)
        self.assertEqual(await p._ask(s, 'material', 'original system', 'original payload', 'low', 10),
                         {'quote': 'source', 'section': 'S'})
        p.llm.chat.assert_awaited_once_with('original system', 'original payload', effort='low',
                                           max_tokens=1500, timeout=13.0, json_mode=True)
        self.assertEqual(s.messages, before)
        self.assertEqual(s.errors, [])

    async def test_guard_and_missing_flag_keep_original_material_budget(self):
        for mode in ('guard', 'missing_flag', 'missing_deadline'):
            p, s = self.make_case(guard=mode == 'guard')
            s.debug = {}
            if mode == 'missing_flag': del s._material_budget_guard
            if mode == 'missing_deadline': del s._source_extract_deadline
            await p._ask(s, 'material', 'system', 'payload', 'low', 10)
            self.assertEqual(p.llm.chat.await_args.kwargs['timeout'], 10)
            p.llm.chat.assert_awaited_once()

    async def test_nonmaterial_real_parent_calls_match_baseline(self):
        for name in ('block', 'other', 'rewrite', 'custom'):
            p, s = self.make_case()
            baseline = Selected.__new__(Selected)
            baseline.llm = SimpleNamespace(chat=AsyncMock(return_value='{}'))
            await p._ask(s, name, 'system', 'payload', 'minimal', 7)
            await baseline._ask(s, name, 'system', 'payload', 'minimal', 7)
            self.assertEqual(p.llm.chat.await_args, baseline.llm.chat.await_args)

    async def test_remaining_deadline_caps_material_timeout(self):
        p, s = self.make_case()
        s._source_extract_deadline = time.monotonic() + 2
        await p._ask(s, 'material', 'system', 'payload', 'low', 10)
        actual = p.llm.chat.await_args.kwargs['timeout']
        self.assertGreater(actual, 1)
        self.assertLessEqual(actual, 2)
        p.llm.chat.assert_awaited_once()

    async def test_expired_request_returns_parent_failure_contract_without_call(self):
        p, s = self.make_case()
        s.errors = ['existing:error']
        s._source_extract_deadline = time.monotonic() - .1
        self.assertEqual(await p._ask(s, 'material', 'system', 'payload', 'low', 10), {})
        self.assertEqual(s.errors, ['existing:error', 'material:TimeoutError'])
        p.llm.chat.assert_not_awaited()

    async def test_provider_timeout_parse_failure_and_cancellation_keep_parent_behavior(self):
        for mode, error in [('timeout', 'material:TimeoutError'), ('parse', 'material:parse'), ('cancel', None)]:
            p, s = self.make_case()
            if mode == 'parse': p.llm.chat.return_value = 'not json'
            else: p.llm.chat.side_effect = TimeoutError() if mode == 'timeout' else asyncio.CancelledError()
            if mode == 'cancel':
                with self.assertRaises(asyncio.CancelledError):
                    await p._ask(s, 'material', 'system', 'payload', 'low', 10)
            else:
                self.assertEqual(await p._ask(s, 'material', 'system', 'payload', 'low', 10), {})
            self.assertEqual(s.errors, [error] if error else [])
            p.llm.chat.assert_awaited_once()

    async def test_actual_prepare_within_passes_sixteen_vs_fourteen_and_preserves_fallback(self):
        for guard in (False, True):
            p, s = self.make_case(guard)
            p.prepare = AsyncMock()
            observed = []
            async def within(instance, coro, seconds, state, name):
                observed.append((seconds, name, instance))
                await coro
                return True
            with patch.object(Selected, '_within', new=within):
                await p._prepare_within(s, extra='original extra')
            self.assertEqual(observed[0][:2], (14.0 if guard else 16.0, 'prepare'))
            self.assertEqual(observed[0][2] is p, guard)
            p.prepare.assert_awaited_once_with(s, 'original extra')
            self.assertEqual(s.section, 'S')
            self.assertEqual(p.prepare_limit, 14.0)

    async def test_missing_flag_prepare_uses_original_budget(self):
        p, s = self.make_case()
        del s._material_budget_guard
        p.prepare = AsyncMock()
        async def within(instance, coro, seconds, state, name):
            self.assertIs(instance, p)
            self.assertEqual(seconds, 14.0)
            await coro
            return True
        with patch.object(Selected, '_within', new=within): await p._prepare_within(s)

    async def test_concurrent_guard_and_tutor_prepare_keep_shared_pipeline_unchanged(self):
        p, tutor = self.make_case()
        guard = p._new_state({'messages': tutor.messages}, True)
        seen = []; entered = asyncio.Event(); release = asyncio.Event()
        async def prepare(instance, state, extra=''):
            seen.append((state._material_budget_guard, instance, instance.prepare_limit))
            if len(seen) == 2: entered.set()
            await release.wait()
            self.assertEqual(p.prepare_limit, 14.0)
            await instance._ask(state, 'material', 'system', 'payload', 'low', 10)
        with patch.object(Selected, 'prepare', new=prepare):
            tasks = [asyncio.create_task(p._prepare_within(s)) for s in (tutor, guard)]
            try:
                await asyncio.wait_for(entered.wait(), 1)
                self.assertEqual(p.prepare_limit, 14.0)
                release.set()
                await asyncio.gather(*tasks)
            finally:
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        for is_guard, instance, limit in seen:
            self.assertEqual(limit, 14.0 if is_guard else 16.0)
            self.assertEqual(instance is p, is_guard)
            self.assertIs(instance.searcher, p.searcher)
            self.assertIs(instance.llm, p.llm)
            self.assertIs(instance._slots, p._slots)
        self.assertCountEqual([a.kwargs['timeout'] for a in p.llm.chat.await_args_list], [10, 13.0])

    async def test_prepare_deadline_fallback_and_cancel_are_unchanged(self):
        p, s = self.make_case()
        p.prepare = AsyncMock()
        async def expired(coro, timeout):
            self.assertEqual(timeout, 16.0)
            coro.close()
            raise asyncio.TimeoutError()
        with patch('asyncio.wait_for', side_effect=expired): await p._prepare_within(s)
        self.assertEqual(s.errors, ['prepare:Deadline'])
        self.assertEqual(s.section, 'S')
        p, s = self.make_case()
        p.prepare = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError): await p._prepare_within(s)
        self.assertEqual(s.errors, [])

    async def test_outer_request_still_uses_twenty_six_second_deadline_and_fallback(self):
        p, _ = self.make_case()
        async def expired(coro, timeout):
            self.assertEqual(timeout, 26)
            coro.close()
            raise asyncio.TimeoutError()
        with patch('asyncio.wait_for', side_effect=expired):
            state = await p._run({'messages': [{'role': 'user', 'content': 'q'}]}, False)
        self.assertEqual(state.errors, ['request:Deadline'])
        self.assertEqual(state.answer, FALLBACK_ANSWER)
        self.assertEqual(state.citations, [])

    def test_only_requested_budget_paths_change(self):
        self.assertEqual(Pipeline.deadline, 26)
        self.assertEqual(Pipeline.source_extract_timeout, 8)
        self.assertEqual(Pipeline.prepare_limit, 14.0)
        for method in ('guard', '_run', '_classify', '_generate', '_check_output', '_collect_checks', '_search', '_candidate_sections'):
            self.assertIs(getattr(Pipeline, method), getattr(Selected, method))
        for name in ('short_votes', 'votes', 'block_min_votes', 'learned_gate_threshold', 'learned_section_threshold', 'tutor_section_threshold'):
            self.assertEqual(getattr(Pipeline, name), getattr(Selected, name))


if __name__ == '__main__': unittest.main()
