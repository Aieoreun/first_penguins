"""Async integration tests for reliable BLOCK-only grounding cancellation."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.peace_cancel_ground import FastCancelBlock
from pipelines.kimjaehak.peace_gate_threshold import FastGate90
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.precise_output import POLICY_HINTS


class CancelGroundTests(unittest.IsolatedAsyncioTestCase):
    def body(self, candidate='도울 수는 없어요. 대신 안전한 학습을 안내할게요.'):
        body = {'messages': [{'role': 'user', 'content': '학습 질문'}]}
        if candidate is not None:
            body['candidate_answer'] = candidate
        return body

    def pipeline(self, cls=FastCancelBlock):
        p = cls.__new__(cls)
        p._slots = asyncio.Semaphore(64)
        p.searcher = Searcher([Chunk('A', '교재 근거')])
        p.span_tagger = SimpleNamespace(predict=lambda text: ([], 0))
        p.llm = SimpleNamespace(chat=AsyncMock())
        async def search(state, extra=''):
            state.chunks = p.searcher.chunks
        p._search = search
        return p

    async def test_late_block_drains_real_five_votes_and_keeps_real_safety(self):
        p = self.pipeline()
        ground_started, ground_drained = asyncio.Event(), asyncio.Event()
        ground_calls = 0
        async def classify(state):
            await ground_started.wait()
            state.label = Label.OFF_TOPIC
        async def chat(system, payload, **kwargs):
            nonlocal ground_calls
            if kwargs['max_tokens'] == 1800:
                await ground_drained.wait()
                self.assertEqual(system, p.policy_system + '\n' + POLICY_HINTS['off_topic'])
                self.assertEqual(kwargs['timeout'], 8)
                return '{"safe":true}'
            ground_calls += 1
            self.assertEqual(kwargs['max_tokens'], 1500)
            if ground_calls < 5:
                return '{"unsupported":["안전한 학습"]}'
            ground_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ground_drained.set()
        p._classify, p.llm.chat.side_effect = classify, chat
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertTrue(ground_drained.is_set())
        self.assertEqual(ground_calls, 5)
        self.assertEqual(p.llm.chat.await_count, 6)
        self.assertEqual(result.debug['fusion_completed_ground_votes'], 4)
        self.assertTrue(result.debug['fusion_ground_cancelled_block'])
        self.assertTrue(result.safe)
        self.assertTrue(result.grounded)
        self.assertEqual(result.unsupported, [])
        self.assertEqual(result.errors, [])
        self.assertEqual(p._slots._value, 64)

    async def test_allow_unknown_or_unreliable_block_does_not_cancel(self):
        for mode in ('allow', 'unknown', 'errors', 'exception', 'timeout'):
            with self.subTest(mode=mode):
                p = self.pipeline()
                started, input_done = asyncio.Event(), asyncio.Event()
                cancelled = []
                if mode == 'timeout':
                    p.guard_prepare_limit = .03
                async def classify(state):
                    await started.wait()
                    state.label = Label.NORMAL if mode == 'allow' else (
                        None if mode == 'unknown' else Label.OFF_TOPIC)
                    try:
                        if mode == 'errors':
                            state.errors.append('material:TimeoutError')
                        elif mode == 'exception':
                            raise ValueError('mock input failure')
                        elif mode == 'timeout':
                            await asyncio.Event().wait()
                    finally:
                        input_done.set()
                async def ground(state):
                    started.set()
                    try:
                        await input_done.wait()
                        await asyncio.sleep(.01)
                        state.grounded, state.unsupported = False, ['fact']
                    except asyncio.CancelledError:
                        cancelled.append(True)
                        raise
                p._classify, p._fusion_ground = classify, ground
                p._fusion_safety = AsyncMock()
                result = await asyncio.wait_for(p.guard(self.body()), 1)
                self.assertEqual(cancelled, [])
                self.assertFalse(result.debug['fusion_ground_cancelled_block'])
                p._fusion_safety.assert_awaited_once()
                if mode == 'allow':
                    self.assertFalse(result.grounded)

    async def test_completed_ground_is_not_cancelled_after_late_block(self):
        p = self.pipeline()
        finished = asyncio.Event()
        async def ground(state):
            state.grounded, state.unsupported = False, ['fact']
            finished.set()
        async def classify(state):
            await finished.wait()
            await asyncio.sleep(.01)
            state.label = Label.OFF_TOPIC
        p._classify, p._fusion_ground = classify, ground
        p._fusion_safety = AsyncMock()
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertFalse(result.debug['fusion_ground_cancelled_block'])
        self.assertTrue(result.grounded)
        self.assertEqual(result.unsupported, [])

    async def test_default_flag_keeps_waiting_for_ground_and_preserves_call_order(self):
        p = self.pipeline(FastGate90)
        started, release = asyncio.Event(), asyncio.Event()
        events = []
        async def classify(state):
            events.append('input-start')
            await started.wait()
            state.label = Label.OFF_TOPIC
            events.append('input-done')
        async def ground(state):
            events.append('ground-start')
            started.set()
            await release.wait()
            events.append('ground-done')
        async def safety(state):
            events.append('safety')
            await asyncio.sleep(.01)
            release.set()
            state.safe = True
        p._classify, p._fusion_ground, p._fusion_safety = classify, ground, safety
        p._ground_until_input = AsyncMock(side_effect=AssertionError('disabled path'))
        result = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertEqual(events, ['input-start', 'ground-start', 'input-done', 'safety', 'ground-done'])
        self.assertNotIn('fusion_ground_cancelled_block', result.debug)
        self.assertTrue(result.safe)
        p._ground_until_input.assert_not_awaited()

    async def test_no_candidate_and_immediate_block_never_start_ground(self):
        for candidate in (None, '도울 수는 없어요.'):
            p = self.pipeline()
            async def classify(state):
                state.label = Label.OFF_TOPIC
            p._classify = classify
            p._fusion_ground, p._fusion_safety = AsyncMock(), AsyncMock()
            p._ground_until_input = AsyncMock()
            result = await p.guard(self.body(candidate))
            p._fusion_ground.assert_not_awaited()
            p._ground_until_input.assert_not_awaited()
            if candidate is None:
                p._fusion_safety.assert_not_awaited()
            else:
                p._fusion_safety.assert_awaited_once()
                self.assertTrue(result.debug['fusion_ground_skipped_fast_block'])
                self.assertFalse(result.debug['fusion_ground_cancelled_block'])

    async def test_external_cancel_drains_pending_input_and_ground(self):
        p = self.pipeline()
        started, input_drained, ground_drained = (asyncio.Event() for _ in range(3))
        async def classify(state):
            try:
                await asyncio.Event().wait()
            finally:
                input_drained.set()
        async def ground(state):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ground_drained.set()
        p._classify, p._fusion_ground = classify, ground
        task = asyncio.create_task(p.guard(self.body()))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(input_drained.is_set())
        self.assertTrue(ground_drained.is_set())
        self.assertEqual(p._slots._value, 64)

    async def test_external_cancel_after_intentional_cancel_still_cancels_safety(self):
        p = self.pipeline()
        started, ground_drained, safety_started, safety_drained = (asyncio.Event() for _ in range(4))
        async def classify(state):
            await started.wait()
            state.label = Label.OFF_TOPIC
        async def ground(state):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ground_drained.set()
        async def safety(state):
            await ground_drained.wait()
            safety_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                safety_drained.set()
        p._classify, p._fusion_ground, p._fusion_safety = classify, ground, safety
        task = asyncio.create_task(p.guard(self.body()))
        await asyncio.wait_for(safety_started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(safety_drained.is_set())
        self.assertEqual(p._slots._value, 64)

    async def test_ground_timeout_and_unexpected_child_cancel_are_not_intentional(self):
        for mode in ('timeout', 'cancel'):
            p = self.pipeline()
            p.guard_ground_timeout = .02
            async def classify(state):
                state.label = Label.NORMAL
            async def ground(state):
                if mode == 'cancel':
                    raise asyncio.CancelledError()
                await asyncio.Event().wait()
            p._classify, p._fusion_ground = classify, ground
            if mode == 'cancel':
                with self.assertRaises(asyncio.CancelledError):
                    await p.guard(self.body())
            else:
                result = await p.guard(self.body())
                self.assertIn('ground:Deadline', result.errors)
                self.assertFalse(result.debug['fusion_ground_cancelled_block'])
            self.assertEqual(p._slots._value, 64)

    async def test_other_task_exception_propagates_and_drains_ground(self):
        p = self.pipeline()
        started, drained = asyncio.Event(), asyncio.Event()
        async def classify(state):
            await started.wait()
            state.label = Label.NORMAL
        async def ground(state):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        async def safety(*args):
            await started.wait()
            raise RuntimeError('outside stage wrapper')
        p._classify, p._fusion_ground, p._run_safety = classify, ground, safety
        with self.assertRaises(RuntimeError):
            await p.guard(self.body())
        self.assertTrue(drained.is_set())
        self.assertEqual(p._slots._value, 64)

    async def test_outer_deadline_cleans_up_pending_checks(self):
        p = self.pipeline()
        p.guard_deadline = .03
        drained = []
        async def classify(state):
            state.label = Label.NORMAL
        async def hang(name):
            try:
                await asyncio.Event().wait()
            finally:
                drained.append(name)
        p._classify = classify
        p._fusion_ground = lambda state: hang('ground')
        p._fusion_safety = lambda state: hang('safety')
        result = await p.guard(self.body())
        self.assertIn('request:Deadline', result.errors)
        self.assertCountEqual(drained, ['ground', 'safety'])
        self.assertFalse(result.debug['fusion_ground_cancelled_block'])
        self.assertEqual(p._slots._value, 64)

    async def test_concurrent_allow_and_block_keep_cancellation_request_local(self):
        p = self.pipeline()
        starts = {name: asyncio.Event() for name in ('block', 'allow')}
        block_drained = asyncio.Event()
        async def classify(state):
            await starts[state.last_user].wait()
            state.label = Label.OFF_TOPIC if state.last_user == 'block' else Label.NORMAL
        async def ground(state):
            starts[state.last_user].set()
            if state.last_user == 'block':
                try:
                    await asyncio.Event().wait()
                finally:
                    block_drained.set()
            else:
                await block_drained.wait()
                state.grounded, state.unsupported = False, ['fact']
        async def safety(state):
            state.safe = True
        p._classify, p._fusion_ground, p._fusion_safety = classify, ground, safety
        bodies = [self.body(), self.body()]
        for body, name in zip(bodies, ('block', 'allow')):
            body['messages'][0]['content'] = name
        blocked, allowed = await asyncio.wait_for(
            asyncio.gather(*(p.guard(body) for body in bodies)), 1)
        self.assertTrue(blocked.debug['fusion_ground_cancelled_block'])
        self.assertFalse(allowed.debug['fusion_ground_cancelled_block'])
        self.assertTrue(blocked.grounded)
        self.assertEqual(blocked.unsupported, [])
        self.assertFalse(allowed.grounded)
        self.assertEqual(allowed.unsupported, ['fact'])
        self.assertEqual(p._slots._value, 64)

    def test_flag_is_only_candidate_change(self):
        self.assertFalse(Fusion.guard_cancel_ground_after_block)
        self.assertFalse(FastGate90.guard_cancel_ground_after_block)
        self.assertTrue(FastCancelBlock.guard_cancel_ground_after_block)
        for name in ('__init__', 'tutor', '_classify', '_fusion_safety', '_collect_checks',
                     '_fusion_ground', '_generate', '_prepare_within'):
            self.assertIs(getattr(FastCancelBlock, name), getattr(FastGate90, name))
        for name in ('guard_deadline', 'guard_prepare_limit', 'guard_ground_timeout',
                     'guard_safety_timeout', 'request_slots', 'guard_gate_threshold'):
            self.assertEqual(getattr(FastCancelBlock, name), getattr(FastGate90, name))


if __name__ == '__main__':
    unittest.main()
