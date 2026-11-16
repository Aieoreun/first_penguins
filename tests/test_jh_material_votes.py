import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.material_votes import Pipeline, select_material_vote


class MaterialVoteSelectionTests(unittest.TestCase):
    def test_majority_returns_first_winning_dict_by_identity(self):
        supported = {'quote': '원문 근거', 'section': 'PY-01-01', 'extra': {'keep': True}}
        unsupported = {'quote': ''}
        later = {'quote': '다른 근거', 'section': 'PY-02-01'}
        self.assertIs(select_material_vote([unsupported, supported, later]), supported)
        self.assertEqual(supported['extra'], {'keep': True})

    def test_whitespace_quotes_vote_unsupported(self):
        first = {'quote': ' \n\t\u3000 ', 'section': 'FIN-01-01'}
        self.assertIs(select_material_vote([{'quote': '근거'}, first, {'quote': ''}]), first)

    def test_tie_uses_first_valid_direction(self):
        yes, no = {'quote': '근거'}, {'quote': ''}
        self.assertIs(select_material_vote([{}, yes, no]), yes)
        self.assertIs(select_material_vote([no, {'quote': None}, yes]), no)

    def test_invalid_types_do_not_vote_and_one_success_is_retained(self):
        valid = {'quote': '근거'}
        for invalid in (None, [], 'quote', {}, {'quote': False}, {'quote': 0}, {'quote': []}):
            with self.subTest(invalid=invalid):
                self.assertIs(select_material_vote([invalid, valid, invalid]), valid)

    def test_no_valid_votes_returns_existing_empty_fallback(self):
        self.assertEqual(select_material_vote([{}, {'quote': None}, None]), {})


class MaterialVotePipelineTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self, guard=False):
        pipeline = Pipeline.__new__(Pipeline)
        state = pipeline._new_state({'messages': [{'role': 'user', 'content': '질문'}]}, guard)
        return pipeline, state

    async def test_guard_and_nonmaterial_calls_are_identical_single_calls(self):
        for guard in (False, True):
            for name in ('block', 'other', 'rewrite', 'material'):
                if not guard and name == 'material':
                    continue
                with self.subTest(guard=guard, name=name):
                    pipeline, state = self.make_case(guard)
                    state.debug = {}  # Endpoint marker must survive replacement.
                    response = {'unvalidated': 'keep original behavior'}
                    with patch.object(ContextFast, '_ask', new_callable=AsyncMock, return_value=response) as original:
                        result = await pipeline._ask(state, name, 'system', 'user', 'low', 10)
                    original.assert_awaited_once_with(state, name, 'system', 'user', 'low', 10)
                    self.assertIs(result, response)

    async def test_tutor_has_exactly_three_calls_with_original_arguments(self):
        pipeline, state = self.make_case()
        response = {'quote': '근거'}
        with patch.object(ContextFast, '_ask', new_callable=AsyncMock, return_value=response) as original:
            result = await pipeline._ask(state, 'material', 'system', 'user', 'low', 10)
        self.assertEqual(original.await_args_list, [call(state, 'material', 'system', 'user', 'low', 10)] * 3)
        self.assertIs(result, response)

    async def test_votes_start_in_parallel_and_tie_ignores_completion_order(self):
        pipeline, state = self.make_case()
        started = asyncio.Event()
        release = [asyncio.Event() for _ in range(3)]
        replies = [{'quote': 'first supported'}, {'quote': ''}, {}]
        entered = []

        async def original(*args):
            index = len(entered)
            entered.append(index)
            if len(entered) == 3:
                started.set()
            await release[index].wait()
            return replies[index]

        with patch.object(ContextFast, '_ask', new=original):
            task = asyncio.create_task(pipeline._ask(state, 'material', 'system', 'user', 'low', 10))
            try:
                await asyncio.wait_for(started.wait(), 1)
                for index in (2, 1, 0):
                    release[index].set()
                    await asyncio.sleep(0)
                self.assertIs(await task, replies[0])
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_real_parent_preserves_error_records_and_votes_only_successes(self):
        pipeline, state = self.make_case()
        state.errors.append('existing:error')
        pipeline.llm = SimpleNamespace(chat=AsyncMock(side_effect=[
            TimeoutError(), '{invalid json', '{"quote":"valid support","section":"PY-01-01"}']))
        result = await pipeline._ask(state, 'material', 'system', 'user', 'low', 10)
        self.assertEqual(result, {'quote': 'valid support', 'section': 'PY-01-01'})
        self.assertEqual(state.errors[0], 'existing:error')
        self.assertCountEqual(state.errors[1:], ['material:TimeoutError', 'material:parse'])
        self.assertEqual(pipeline.llm.chat.await_count, 3)
        for args in pipeline.llm.chat.await_args_list:
            self.assertEqual(args, call('system', 'user', effort='low', max_tokens=1500, timeout=10, json_mode=True))

    async def test_all_failed_parent_calls_keep_fallback_without_retry(self):
        pipeline, state = self.make_case()
        pipeline.llm = SimpleNamespace(chat=AsyncMock(side_effect=TimeoutError))
        self.assertEqual(await pipeline._ask(state, 'material', 'system', 'user', 'low', 10), {})
        self.assertEqual(state.errors, ['material:TimeoutError'] * 3)
        self.assertEqual(pipeline.llm.chat.await_count, 3)

    async def test_parent_cancellation_reaches_all_three_children(self):
        pipeline, state = self.make_case()
        started = asyncio.Event()
        entered, cancelled = [], []

        async def chat(*args, **kwargs):
            index = len(entered)
            entered.append(index)
            if len(entered) == 3:
                started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(index)
                raise

        pipeline.llm = SimpleNamespace(chat=chat)
        task = asyncio.create_task(pipeline._ask(state, 'material', 'system', 'user', 'low', 10))
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertCountEqual(cancelled, [0, 1, 2])
        self.assertEqual(state.errors, [])

    async def test_child_cancellation_stops_siblings_and_propagates(self):
        pipeline, state = self.make_case()
        started = asyncio.Event()
        entered, finished = [], []

        async def chat(*args, **kwargs):
            index = len(entered)
            entered.append(index)
            if len(entered) == 3:
                started.set()
            try:
                await started.wait()
                if index == 0:
                    raise asyncio.CancelledError()
                await asyncio.Event().wait()
            finally:
                finished.append(index)

        pipeline.llm = SimpleNamespace(chat=chat)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(pipeline._ask(state, 'material', 'system', 'user', 'low', 10), 1)
        self.assertCountEqual(finished, [0, 1, 2])
        self.assertEqual(state.errors, [])

    async def test_guard_marker_is_request_local_on_shared_pipeline(self):
        pipeline, tutor = self.make_case()
        guard = pipeline._new_state({'messages': [{'role': 'user', 'content': 'guard'}]}, True)
        tutor.debug = guard.debug = {}
        seen = []

        async def original(self, state, *args):
            seen.append(state)
            await asyncio.sleep(0)
            return {'quote': state.last_user}

        with patch.object(ContextFast, '_ask', new=original):
            results = await asyncio.gather(
                pipeline._ask(tutor, 'material', 'system', 'user', 'low', 10),
                pipeline._ask(guard, 'material', 'system', 'user', 'low', 10))
        self.assertEqual(sum(state is tutor for state in seen), 3)
        self.assertEqual(sum(state is guard for state in seen), 1)
        self.assertEqual(results, [{'quote': '질문'}, {'quote': 'guard'}])

    def test_existing_budget_and_other_methods_remain_inherited(self):
        from pipelines import load
        self.assertIs(load('jh_r3_material_vote3'), Pipeline)
        for name in ('guard', '_collect_checks', '_prepare_within', '_within', '_search', '_classify', '_generate', '_check_output'):
            self.assertIs(getattr(Pipeline, name), getattr(ContextFast, name), name)
        self.assertEqual(Pipeline.prepare_limit, ContextFast.prepare_limit)
        self.assertEqual(Pipeline.deadline, ContextFast.deadline)


if __name__ == '__main__':
    unittest.main()
