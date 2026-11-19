import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.budget_stream import Pipeline
from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.material_budget import Pipeline as MaterialBudget
from pipelines.kimjaehak.source_stream import Pipeline as SourceStream
from tests.test_jh_source_stream import Stream, WAIT


class BudgetStreamTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self, parts=None):
        pipeline = Pipeline.__new__(Pipeline)
        chunks = [Chunk('S' + str(i), 'Lesson ' + str(i) + '\nmaterial ' + str(i))
                  for i in range(8)]
        pipeline.searcher = Searcher(chunks)
        pipeline._slots = asyncio.Semaphore(2)
        stream = Stream(parts if parts is not None else ['{"section":', '"S1",', WAIT])
        create = AsyncMock(return_value=stream)
        pipeline.llm = SimpleNamespace(
            model='test-model',
            chat=AsyncMock(return_value='{"quote":"material","section":"S0"}'),
            _chat=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        state = pipeline._new_state({'messages': [{'role': 'user', 'content': 'material 1'}]}, False)
        state.label, state.section, state.chunks = Label.NORMAL, 'S0', chunks
        return pipeline, state, stream, create

    async def test_concurrent_prepare_clones_keep_stream_mro_and_guard_isolation(self):
        pipeline, tutor, stream, create = self.make_case()
        guard = pipeline._new_state({'messages': tutor.messages}, True)
        guard.chunks = tutor.chunks
        entered, release = asyncio.Event(), asyncio.Event()
        seen, limits = [], []
        original_within = SelectedFocused._within

        async def prepare(instance, state, extra=''):
            seen.append((state._material_budget_guard, instance, instance.prepare_limit))
            if len(seen) == 2:
                entered.set()
            await release.wait()
            result = await instance._ask(state, 'material', 'material system',
                                         'material payload', 'low', 10)
            self.assertEqual(result['quote'], 'material')
            # The tutor request copy retains both parents, including streaming.
            self.assertIs(instance._source_response.__func__, SourceStream._source_response)

        async def within(instance, coro, seconds, state, name):
            limits.append((state._material_budget_guard, seconds))
            return await original_within(instance, coro, seconds, state, name)

        with patch.object(SelectedFocused, 'prepare', new=prepare), \
                patch.object(SelectedFocused, '_within', new=within):
            tasks = [asyncio.create_task(pipeline._prepare_within(s)) for s in (tutor, guard)]
            try:
                await asyncio.wait_for(entered.wait(), 1)
                self.assertEqual(pipeline.prepare_limit, 14)
                release.set()
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        self.assertCountEqual(limits, [(False, 16), (True, 14)])
        for is_guard, instance, limit in seen:
            self.assertEqual(instance is pipeline, is_guard)
            self.assertEqual(limit, 14 if is_guard else 16)
            self.assertIs(instance.llm, pipeline.llm)
            self.assertIs(instance.searcher, pipeline.searcher)
            self.assertIs(instance._slots, pipeline._slots)
        self.assertCountEqual([c.kwargs['timeout'] for c in pipeline.llm.chat.await_args_list], [13, 10])
        self.assertTrue(all(c.kwargs['max_tokens'] == 1500 for c in pipeline.llm.chat.await_args_list))

        await pipeline._generate(tutor)
        self.assertEqual(tutor.section, 'S1')
        self.assertEqual(tutor.citations, ['S1'])
        self.assertIn('material 1', tutor.answer)
        self.assertTrue(stream.closed)
        self.assertEqual(stream.read, 2)  # The reason tail was not consumed.
        create.assert_awaited_once()
        self.assertEqual(pipeline.llm.chat.await_count, 2)  # Material only, no second source call.
        kwargs = create.await_args.kwargs
        self.assertEqual(kwargs['max_completion_tokens'], 2000)
        self.assertEqual(kwargs['extra_body'], {'reasoning_effort': 'low'})
        self.assertEqual(kwargs['response_format'], {'type': 'json_object'})
        self.assertTrue(kwargs['stream'])
        payload = json.loads(kwargs['messages'][1]['content'])
        self.assertEqual(payload, pipeline._source_payload(tutor, ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']))
        self.assertEqual(len(payload['materials']), 6)
        self.assertEqual(pipeline.prepare_limit, 14)

    async def test_material_and_source_share_remaining_request_budget(self):
        pipeline, state, stream, create = self.make_case()

        async def material(*args, **kwargs):
            self.assertEqual(kwargs['timeout'], 13)
            # Simulate earlier stages consuming the request budget without sleep.
            state._source_extract_deadline = time.monotonic() + 2
            return '{"quote":"material"}'

        pipeline.llm.chat.side_effect = material
        await pipeline._ask(state, 'material', 'system', 'payload', 'low', 10)
        with patch.object(pipeline, '_source_response', wraps=pipeline._source_response) as response:
            await pipeline._generate(state)
        timeout = response.await_args.kwargs['timeout']
        self.assertGreater(timeout, 1)
        self.assertLessEqual(timeout, 1.5)
        self.assertEqual(state.section, 'S1')
        self.assertTrue(stream.closed)
        self.assertEqual(pipeline.llm.chat.await_count, 1)
        self.assertEqual(create.await_count, 1)

    async def test_source_error_after_material_keeps_excerpt_without_retry(self):
        pipeline, state, stream, create = self.make_case([RuntimeError('transport')])
        await pipeline._ask(state, 'material', 'system', 'payload', 'low', 10)
        await pipeline._generate(state)
        self.assertEqual(state.section, 'S0')
        self.assertEqual(state.citations, ['S0'])
        self.assertEqual(state.errors, ['source_extract:RuntimeError'])
        self.assertTrue(stream.closed)
        self.assertEqual(pipeline.llm.chat.await_count, 1)
        self.assertEqual(create.await_count, 1)

    async def test_cancel_during_source_after_material_closes_and_propagates(self):
        pipeline, state, stream, create = self.make_case([WAIT])
        await pipeline._ask(state, 'material', 'system', 'payload', 'low', 10)
        task = asyncio.create_task(pipeline._generate(state))
        await stream.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.closed)
        self.assertEqual(state.errors, [])
        self.assertEqual(pipeline.llm.chat.await_count, 1)
        self.assertEqual(create.await_count, 1)

    async def test_guard_and_high_section_gates_still_bypass_stream(self):
        for mode in ('guard', 'learned_section', 'tutor_section_gate'):
            pipeline, state, stream, create = self.make_case()
            if mode == 'guard':
                state = pipeline._new_state({'messages': state.messages}, True)
                state.label, state.section = Label.NORMAL, 'S0'
                await pipeline._ask(state, 'material', 'system', 'payload', 'low', 10)
                self.assertEqual(pipeline.llm.chat.await_args.kwargs['timeout'], 10)
                with patch.object(ContextFast, '_generate', new_callable=AsyncMock) as original:
                    await pipeline._generate(state)
                    original.assert_awaited_once_with(state)
            else:
                state.debug[mode] = True
                await pipeline._generate(state)
                self.assertEqual(state.citations, ['S0'])
            create.assert_not_awaited()
            self.assertFalse(stream.closed)  # No handle acquired.

    def test_composition_inherits_only_existing_behaviors_and_metadata(self):
        self.assertEqual(Pipeline.__bases__, (MaterialBudget, SourceStream))
        self.assertIs(Pipeline._ask, MaterialBudget._ask)
        self.assertIs(Pipeline._prepare_within, MaterialBudget._prepare_within)
        self.assertIs(Pipeline._source_response, SourceStream._source_response)
        for name in ('guard', '_run', '_classify', '_generate', '_source_candidates', '_candidate_sections'):
            self.assertIs(getattr(Pipeline, name), getattr(SelectedFocused, name))
        self.assertEqual((Pipeline.tutor_material_timeout, Pipeline.tutor_prepare_limit,
                          Pipeline.prepare_limit, Pipeline.source_extract_timeout,
                          Pipeline.deadline, Pipeline.source_extract_sections),
                         (13, 16, 14, 8, 26, 6))
        self.assertTrue(Pipeline.source_stream_early_finish)
        self.assertTrue(Pipeline.source_extraction_selector)
        self.assertEqual(Pipeline.source_stream_buffer_chars, 16000)
        for name in ('learned_gate_threshold', 'learned_section_threshold', 'tutor_section_threshold',
                     'votes', 'short_votes', 'block_min_votes', 'tutor_extract_scope'):
            self.assertEqual(getattr(Pipeline, name), getattr(SelectedFocused, name))


if __name__ == '__main__':
    unittest.main()
