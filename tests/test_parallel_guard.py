"""Contracts for independent, concurrently executed guard branches."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.parallel_guard import ParallelGuard, Pipeline, SAFETY_SYSTEM
from pipelines.kimjaehak.material_budget import Pipeline as MaterialBudget


class Fake(ParallelGuard):
    deadline = 1.0
    prepare_limit = 0.7
    guard_prepare_limit = 0.7
    output_votes = 5

    def __init__(self):
        self._slots = asyncio.Semaphore(2)
        self.events = []
        self.started = set()
        self.all_started = asyncio.Event()
        self.ground_state = None
        self.safety_state = None
        self.label = Label.NORMAL

    def _new_state(self, body, guard):
        return State.from_request(body)

    async def _within(self, coro, seconds, state, name):
        try:
            await asyncio.wait_for(coro, timeout=max(.001, seconds))
            return True
        except asyncio.TimeoutError:
            state.errors.append(f'{name}:Deadline')
            return False

    async def _search(self, state, extra=''):
        self.events.append(('search', extra))
        state.chunks = [Chunk('S', 'support')]

    async def _sync(self, name):
        self.started.add(name)
        if len(self.started) == 3:
            self.all_started.set()
        await asyncio.wait_for(self.all_started.wait(), .3)

    async def _classify(self, state):
        self.events.append(('classify', state.chunks[0].section_id))
        await self._sync('classify')
        state.label = self.label
        state.section = 'S'

    async def _parallel_ground(self, state):
        self.events.append(('ground', state.chunks[0].section_id))
        self.ground_state = state
        await self._sync('ground')
        state.grounded = False
        state.unsupported = ['claim']

    async def _parallel_safety(self, state):
        self.events.append(('safety', state.candidate_answer))
        self.safety_state = state
        await self._sync('safety')
        state.safe = False


class ParallelGuardTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def body(candidate=True):
        value = {'messages': [{'role': 'user', 'content': 'Explain'}]}
        if candidate is not None:
            value['candidate_answer'] = 'claim' if candidate is True else candidate
        return value

    def test_tutor_still_uses_original_implementation(self):
        self.assertIs(Pipeline.tutor, MaterialBudget.tutor)

    async def test_search_precedes_three_concurrent_branches_and_results_merge(self):
        p = Fake()
        for label in (Label.NORMAL, Label.HARMFUL_REQUEST):
            p.label = label
            p.started = set()
            p.all_started = asyncio.Event()
            p.events = []
            result = await p.guard(self.body())
            self.assertEqual(p.events[0], ('search', 'claim'))
            self.assertEqual(p.started, {'classify', 'ground', 'safety'})
            self.assertEqual(result.label, label)
            self.assertEqual(result.grounded, label is Label.HARMFUL_REQUEST)
            self.assertEqual(result.unsupported, [] if label is Label.HARMFUL_REQUEST else ['claim'])
            self.assertFalse(result.safe)
            self.assertIsNot(p.ground_state, p.safety_state)
            self.assertIsNot(p.ground_state.errors, p.safety_state.errors)
            self.assertEqual(result.to_guard_response()['output']['unsupported'],
                             [] if label is Label.HARMFUL_REQUEST else ['claim'])

    async def test_missing_candidate_skips_both_output_checks(self):
        p = Fake()
        async def classify(state):
            state.label = Label.NORMAL
            state.section = 'S'
        p._classify = AsyncMock(side_effect=classify)
        p._parallel_ground = AsyncMock(side_effect=AssertionError('unexpected grounding'))
        p._parallel_safety = AsyncMock(side_effect=AssertionError('unexpected safety'))
        result = await p.guard(self.body(None))
        p._classify.assert_awaited_once()
        p._parallel_ground.assert_not_awaited()
        p._parallel_safety.assert_not_awaited()
        self.assertNotIn('output', result.to_guard_response())

    async def test_empty_candidate_is_still_a_candidate(self):
        p = Fake()
        result = await p.guard(self.body(''))
        self.assertEqual(p.started, {'classify', 'ground', 'safety'})
        self.assertIn('output', result.to_guard_response())

    async def test_external_cancellation_stops_all_branches(self):
        p = Fake()
        started = set()
        stopped = set()
        all_started = asyncio.Event()
        def branch(name):
            async def go(state):
                started.add(name)
                if len(started) == 3:
                    all_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.add(name)
            return go
        p._classify = branch('classify')
        p._parallel_ground = branch('ground')
        p._parallel_safety = branch('safety')
        task = asyncio.create_task(p.guard(self.body()))
        await asyncio.wait_for(all_started.wait(), .5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(stopped, started)
        self.assertEqual(p._slots._value, 2)

    async def test_request_deadline_cancels_pending_work(self):
        p = Fake()
        p.deadline = .08
        p.prepare_limit = .07
        p.guard_prepare_limit = .07
        cancelled = []
        async def blocked(state):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
        p._classify = blocked
        p._parallel_ground = blocked
        p._parallel_safety = blocked
        result = await asyncio.wait_for(p.guard(self.body()), .5)
        self.assertTrue(any('Deadline' in e for e in result.errors), result.errors)
        self.assertEqual(len(cancelled), 3)
        self.assertEqual(p._slots._value, 2)

    async def test_safety_sees_only_dialogue_and_candidate_and_checks_boolean(self):
        p = Fake()
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"safe":true}'))
        state = State.from_request(self.body())
        state.label = Label.HARMFUL_REQUEST
        await ParallelGuard._parallel_safety(p, state)
        self.assertTrue(state.safe)
        args = p.llm.chat.await_args
        self.assertEqual(args.args[0], SAFETY_SYSTEM)
        payload = json.loads(args.args[1])
        self.assertEqual(set(payload), {'conversation', 'candidate_answer'})
        self.assertEqual(payload['candidate_answer'], 'claim')
        p.llm.chat.return_value = '{"safe":"true"}'
        await ParallelGuard._parallel_safety(p, state)
        self.assertFalse(state.safe)
        self.assertIn('independent_safety:ValueError', state.errors)

    async def test_grounding_uses_five_independent_votes(self):
        p = Fake()
        state = State.from_request(self.body())
        calls = []
        async def ground(_, trial):
            calls.append(trial)
            trial.unsupported = ['claim'] if len(calls) <= 3 else []
        with patch('pipelines.kimjaehak.parallel_guard.ContextOutput._check_output', new=ground):
            await ParallelGuard._parallel_ground(p, state)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len({id(c.errors) for c in calls}), 5)
        self.assertFalse(state.grounded)
        self.assertEqual(state.unsupported, ['claim'])


if __name__ == '__main__':
    unittest.main()
