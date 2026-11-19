"""Guard-only ablations: ordering, cancellation, real budgets and input bypass."""
import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.classifier_matrix import CharMLP50, MatrixGate
from pipelines.kimjaehak.neural_peace_guard import (
    ParallelSafety, CheckBudgets, PeaceBlock, UnlearnedInput, _guard_deadline)
from pipelines.kimjaehak.peace_guard_fusion import Pipeline as Fusion
from pipelines.kimjaehak.peace_guard_prompt import PEACE_BLOCK_SYSTEM, SAFETY_SYSTEM
from pipelines.kimjaehak.voted import Pipeline as Voted


class GuardTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, cls=ParallelSafety):
        p = cls.__new__(cls)
        p._slots = asyncio.Semaphore(64)
        p.searcher = Searcher([Chunk('A', '교재 근거')])
        p.span_tagger = SimpleNamespace(predict=lambda text: ([], 0))
        p.llm = SimpleNamespace(chat=AsyncMock())
        async def search(state, extra=''):
            state.chunks = p.searcher.chunks
        p._search = search
        return p

    def body(self):
        return {'messages': [{'role':'user','content':'학습 질문'}], 'candidate_answer':'후보 답변'}

    async def test_independent_safety_starts_before_input_even_during_grace(self):
        p = self.pipeline()
        p.guard_fast_input_wait = .2
        started = asyncio.Event()
        async def classify(state):
            await started.wait()
            state.label = Label.NORMAL
        async def chat(system, payload, **kwargs):
            if kwargs['max_tokens'] == 1800:
                self.assertEqual(system, SAFETY_SYSTEM)
                self.assertEqual(json.loads(payload), {'conversation':self.body()['messages'], 'candidate_answer':'후보 답변'})
                started.set()
                return '{"safe":false}'
            return '{"unsupported":[]}'
        p._classify, p.llm.chat.side_effect = classify, chat
        state = await asyncio.wait_for(p.guard(self.body()), .1)
        self.assertEqual(state.label, Label.NORMAL)
        self.assertFalse(state.safe)
        self.assertEqual(state.errors, [])

    async def test_block_cancels_ground_but_keeps_safety_and_finished_votes(self):
        p = self.pipeline()
        ground_started, ground_drained, safety_started = (asyncio.Event() for _ in range(3))
        count = 0
        async def classify(state):
            await ground_started.wait()
            await safety_started.wait()
            state.label = Label.OFF_TOPIC
        async def chat(system, payload, **kwargs):
            nonlocal count
            if kwargs['max_tokens'] == 1800:
                safety_started.set()
                await ground_drained.wait()
                return '{"safe":true}'
            count += 1
            if count < 5: return '{"unsupported":[]}'
            ground_started.set()
            try: await asyncio.Event().wait()
            finally: ground_drained.set()
        p._classify, p.llm.chat.side_effect = classify, chat
        state = await asyncio.wait_for(p.guard(self.body()), 1)
        self.assertTrue(state.safe)
        self.assertTrue(state.debug['fusion_ground_cancelled_block'])
        self.assertEqual(state.debug['fusion_completed_ground_votes'], 4)
        self.assertEqual(state.errors, [])

    async def test_no_candidate_no_safety_and_external_cancel_drains_children(self):
        p = self.pipeline()
        p._classify = AsyncMock()
        await p.guard({'messages':self.body()['messages']})
        p.llm.chat.assert_not_awaited()
        started, drained = asyncio.Event(), asyncio.Event()
        async def chat(*args, **kwargs):
            started.set()
            try: await asyncio.Event().wait()
            finally: drained.set()
        p.llm.chat.side_effect = chat
        task = asyncio.create_task(p.guard(self.body()))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue(drained.is_set())
        self.assertEqual(p._slots._value,64)

    async def test_actual_call_budgets_and_guard_only_prompt(self):
        for cls in (CheckBudgets, PeaceBlock):
            p = self.pipeline(cls)
            p.llm.chat.return_value = '{"category":"none","safe":true}'
            guard, tutor = p._new_state(self.body(),True), p._new_state(self.body(),False)
            token = _guard_deadline.set(time.monotonic()+.5)
            try:
                await p._ask(guard,'material','system','payload','low',10)
                self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'],.5)
                await p._ground_response({})
                self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'],.5)
                await p._fusion_safety(guard)
                self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'],.5)
                await p._ask(tutor,'block','system','payload','low',7)
                self.assertEqual(p.llm.chat.await_args.kwargs['timeout'],7)
                self.assertEqual(p._block_system(tutor), CharMLP50._block_system(p,tutor))
                self.assertEqual(p._block_system(guard), PEACE_BLOCK_SYSTEM if cls is PeaceBlock else CharMLP50._block_system(p,guard))
            finally: _guard_deadline.reset(token)
            await p._ask(guard,'material','system','payload','low',10)
            self.assertEqual(p.llm.chat.await_args.kwargs['timeout'],16)

    async def test_tie_budget_not_reset_and_tutor_group_unchanged(self):
        p = self.pipeline(CheckBudgets)
        seen=[]
        async def ask(state,name,system,check,effort,timeout):
            seen.append(timeout)
            if len(seen) <= 2:
                await asyncio.sleep(.01)
                return {'category':'none' if len(seen)==1 else 'off_topic'}
            return {'category':'none'}
        # Explicit disagreement independent of coroutine completion order.
        async def first(state,name,system,check,effort,timeout):
            idx=len(seen); seen.append(timeout)
            await asyncio.sleep(.01)
            return {'category': 'none' if idx==0 else 'off_topic'}
        p._ask=first
        await p._vote_group(p._new_state(self.body(),True),'block','s','p')
        self.assertEqual(len(seen),3)
        self.assertEqual(seen[:2],[13,13])
        self.assertLess(seen[2],13)
        from pipelines.kimjaehak.adaptive import Pipeline as Adaptive
        with patch.object(Adaptive,'_vote_group',new_callable=AsyncMock) as original:
            state=p._new_state(self.body(),False)
            await p._vote_group(state,'block','s','p')
            original.assert_awaited_once_with(state,'block','s','p')

    async def test_queue_and_search_use_request_budget_without_leaking_to_tutor(self):
        p = self.pipeline(CheckBudgets)
        p.guard_deadline=.08
        p._slots=asyncio.Semaphore(0)
        async def search(state,extra=''):
            await asyncio.sleep(.02)
            state.chunks=p.searcher.chunks
        p._search=search
        budgets=[]
        async def classify(state):
            budgets.append(p._guard_seconds(16))
            state.label=Label.NORMAL
        p._classify=classify
        p.llm.chat.return_value='{"safe":true,"unsupported":[]}'
        task=asyncio.create_task(p.guard(self.body()))
        await asyncio.sleep(.02)
        self.assertIsNone(_guard_deadline.get())
        p._slots.release()
        result=await task
        self.assertLess(budgets[0],.05)
        self.assertIsNone(_guard_deadline.get())
        self.assertEqual(result.errors,[])
        self.assertEqual(p.guard_safety_timeout,13)

    async def test_expired_budget_starts_no_api_call(self):
        p=self.pipeline(CheckBudgets)
        token=_guard_deadline.set(time.monotonic()-1)
        try:
            state=p._new_state(self.body(),True)
            self.assertEqual(await p._ask(state,'material','s','u','low',10),{})
            with self.assertRaises(asyncio.TimeoutError): await p._ground_response({})
            p.llm.chat.assert_not_awaited()
            self.assertEqual(state.errors,['material:NoBudget'])
        finally: _guard_deadline.reset(token)

    async def test_e_bypasses_all_guard_input_models_but_not_tutor(self):
        p=self.pipeline(UnlearnedInput)
        # No fitted input models exist on this instance. Real Voted path works.
        p._collect_checks=AsyncMock(return_value=([{'category':'none'}],[{'category':'none'}],{'quote':'근거'}))
        state=p._new_state(self.body(),True)
        await p._classify(state)
        self.assertEqual(state.label,Label.NORMAL)
        self.assertTrue(state.debug['guard_unlearned_input'])
        tutor=p._new_state(self.body(),False)
        with patch.object(MatrixGate,'_classify',new_callable=AsyncMock) as original:
            await p._classify(tutor)
            original.assert_awaited_once_with(tutor)
        self.assertIs(UnlearnedInput._fusion_ground,CharMLP50._fusion_ground)

    def test_tutor_generation_search_and_public_baseline_preserved(self):
        for cls in (ParallelSafety,CheckBudgets,PeaceBlock,UnlearnedInput):
            for name in ('tutor','_generate','_search','_prepare_within','_check_output','_fusion_ground'):
                self.assertIs(getattr(cls,name),getattr(CharMLP50,name))
        self.assertTrue(CharMLP50.guard_safety_after_input)
        self.assertFalse(CharMLP50.guard_safety_launch_early)
        self.assertEqual(CharMLP50.guard_safety_timeout,8)
        self.assertEqual(CharMLP50.guard_ground_timeout,13)


if __name__=='__main__': unittest.main()
