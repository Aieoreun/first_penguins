"""Offline ALLOW policy check, original BLOCK behavior and deadline contracts."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from core.schema import Label
from core.search import Searcher
from core.state import Chunk, State
from pipelines.kimjaehak.peace_allow_safety import AllowSafety, ALLOW_HINT
from pipelines.kimjaehak.peace_cancel_ground import FastCancelBlock
from pipelines.kimjaehak.precise_output import POLICY_SYSTEM


class AllowSafetyTests(unittest.IsolatedAsyncioTestCase):
    def state(self, candidate='일반 학습 설명', label=Label.NORMAL):
        return State(messages=[{'role':'user','content':'교재 개념을 설명해 주세요.'}],
                     candidate_answer=candidate, label=label, grounded=False, unsupported=['설명'])

    def pipeline(self, cls=AllowSafety, reply='{"safe":true}'):
        p = cls.__new__(cls)
        p.llm = SimpleNamespace(chat=AsyncMock(return_value=reply))
        return p

    async def test_allow_runs_exactly_one_neutral_policy_check_and_preserves_other_fields(self):
        for verdict in (True,False):
            p = self.pipeline(reply=json.dumps({'safe':verdict}))
            state = self.state()
            await p._fusion_safety(state)
            self.assertIs(state.safe,verdict)
            self.assertIs(state.grounded,False)
            self.assertEqual(state.unsupported,['설명'])
            self.assertIs(state.label,Label.NORMAL)
            p.llm.chat.assert_awaited_once()
            args,kwargs = p.llm.chat.call_args
            self.assertEqual(args[0],POLICY_SYSTEM+'\n'+ALLOW_HINT)
            self.assertEqual(json.loads(args[1]),{'conversation':state.messages,'candidate_answer':state.candidate_answer})
            self.assertEqual(kwargs,{'effort':'low','max_tokens':1500,'timeout':8,'json_mode':True})
            self.assertNotIn('input_assessment',json.loads(args[1]))

    async def test_invalid_boolean_provider_failure_and_timeout_keep_original_true(self):
        for reply in ('{}','{"safe":"false"}','{"safe":0}','{"safe":null}','not json'):
            p,state = self.pipeline(reply=reply),self.state()
            state.safe = False
            await p._fusion_safety(state)
            self.assertTrue(state.safe)
            self.assertEqual(state.errors,['allow_safety:ValueError'])
        p,state = self.pipeline(),self.state()
        p.llm.chat.side_effect = RuntimeError('private provider response')
        await p._fusion_safety(state)
        self.assertTrue(state.safe)
        self.assertEqual(state.errors,['allow_safety:RuntimeError'])
        p,state = self.pipeline(),self.state()
        p.guard_allow_safety_timeout = .01
        drained = asyncio.Event()
        async def hang(*args,**kwargs):
            try: await asyncio.Event().wait()
            finally: drained.set()
        p.llm.chat.side_effect = hang
        await asyncio.wait_for(p._fusion_safety(state),.5)
        self.assertTrue(drained.is_set())
        self.assertTrue(state.safe)
        self.assertEqual(state.errors,['allow_safety:TimeoutError'])

    async def test_external_cancellation_propagates_and_drains_single_call(self):
        p,state = self.pipeline(),self.state()
        started,drained = asyncio.Event(),asyncio.Event()
        async def hang(*args,**kwargs):
            started.set()
            try: await asyncio.Event().wait()
            finally: drained.set()
        p.llm.chat.side_effect = hang
        task = asyncio.create_task(p._fusion_safety(state))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue(drained.is_set())
        self.assertEqual(state.errors,[])
        p.llm.chat.assert_awaited_once()

    async def test_actual_block_parent_behavior_payload_and_prompt_are_identical(self):
        for candidate in ('도울 수는 없어요. 대신 학습을 안내해요.','요청하신 내용을 해드릴게요.'):
            for label in (Label.OFF_TOPIC,Label.HARMFUL_REQUEST,Label.JAILBREAK_INJECTION):
                old,new = self.pipeline(FastCancelBlock),self.pipeline()
                a,b = self.state(candidate,label),self.state(candidate,label)
                await old._fusion_safety(a)
                await new._fusion_safety(b)
                self.assertEqual(a.to_guard_response(),b.to_guard_response())
                self.assertEqual(a.errors,b.errors)
                self.assertEqual(old.llm.chat.await_args_list,new.llm.chat.await_args_list)

    async def test_concurrent_allow_requests_do_not_mutate_shared_configuration(self):
        p = self.pipeline()
        before = dict(vars(p))
        async def chat(system,payload,**kwargs):
            await asyncio.sleep(0)
            return json.dumps({'safe':json.loads(payload)['candidate_answer']=='safe fixture'})
        p.llm.chat.side_effect = chat
        states = [self.state('safe fixture'),self.state('unsafe fixture')]
        await asyncio.gather(*(p._fusion_safety(s) for s in states))
        self.assertEqual([s.safe for s in states],[True,False])
        self.assertEqual(vars(p),before)

    async def test_inherited_guard_merges_grounding_and_safety_without_extra_budget(self):
        p = self.pipeline(reply='{"safe":false}')
        p._slots = asyncio.Semaphore(64)
        p.searcher = Searcher([Chunk('S','교재 근거')])
        async def search(state,extra=''): state.chunks = p.searcher.chunks
        async def classify(state): state.label = Label.NORMAL
        async def ground(state): state.grounded,state.unsupported = False,['fixture']
        p._search,p._classify,p._fusion_ground = search,classify,ground
        result = await asyncio.wait_for(p.guard({'messages':[{'role':'user','content':'학습 질문'}],
                                                'candidate_answer':'fixture'}),1)
        self.assertFalse(result.safe)
        self.assertFalse(result.grounded)
        self.assertEqual(result.unsupported,['fixture'])
        self.assertEqual(result.errors,[])
        self.assertEqual(p._slots._value,64)
        p.llm.chat.assert_awaited_once()
        p.llm.chat.reset_mock()
        await p.guard({'messages':[{'role':'user','content':'학습 질문'}]})
        p.llm.chat.assert_not_awaited()

    async def test_outer_remaining_budget_cancels_review_and_preserves_allow_fallback(self):
        p = self.pipeline()
        p.guard_deadline = .04
        started,drained = asyncio.Event(),asyncio.Event()
        async def hang(*args,**kwargs):
            started.set()
            try: await asyncio.Event().wait()
            finally: drained.set()
        p.llm.chat.side_effect = hang
        state,input_state = self.state(),self.state()
        input_task = asyncio.create_task(asyncio.sleep(0,result=True))
        import time
        await asyncio.wait_for(p._run_safety(state,input_state,input_task,time.monotonic()),.5)
        self.assertTrue(started.is_set())
        self.assertTrue(drained.is_set())
        self.assertTrue(state.safe)
        self.assertEqual(state.errors,['safety:Deadline'])

    def test_only_allow_safety_changes_and_guard_deadline_remains_28(self):
        self.assertEqual(AllowSafety.guard_deadline,28)
        self.assertEqual(AllowSafety.guard_safety_timeout,8)
        for method in ('guard','tutor','_classify','_generate','_fusion_ground','_run_safety',
                       '_fusion_post_checks','_ground_until_input','_fusion_safety_payload'):
            self.assertIs(getattr(AllowSafety,method),getattr(FastCancelBlock,method))


if __name__ == '__main__':
    unittest.main()
