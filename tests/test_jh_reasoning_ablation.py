"""Actual stage calls, effort isolation and baseline preservation."""
import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipelines import load
load('jh_cm_char_mlp50')
from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.neural_peace_guard import CheckBudgets, _guard_deadline
from pipelines.kimjaehak.reasoning_ablation import (
    EffortClient, GuardInputMedium, GuardGroundMedium, GuardSafetyMedium,
    TutorInputMedium, TutorSourceMedium, TutorAnswerMedium)
from pipelines.v1.pipeline import Pipeline as V1


class ReasoningTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, cls):
        p=cls.__new__(cls)
        p.llm=SimpleNamespace(chat=AsyncMock(return_value='{"category":"none","quote":"근거","safe":true,"unsupported":[]}'),
            embed=AsyncMock(return_value=[[1.0]]),can_embed=True)
        p._slots=asyncio.Semaphore(64)
        p.searcher=Searcher([Chunk('A','교재 근거')])
        p.span_tagger=SimpleNamespace(predict=lambda text:([],0))
        async def search(state,extra=''):state.chunks=p.searcher.chunks
        p._search=search
        return p

    def state(self,p,guard):
        state=p._new_state({'messages':[{'role':'user','content':'교재 설명 부탁해요'}],
                            'candidate_answer':'후보 답변'},guard)
        state.section='A'; state.chunks=p.searcher.chunks
        return state

    async def test_input_effort_only_changes_target_endpoint_and_keeps_call_contract(self):
        for cls,base,target_guard in ((GuardInputMedium,CheckBudgets,True),(TutorInputMedium,CharMLP50,False)):
            for guard in (True,False):
                for name in ('block','other','material'):
                    p,q=self.pipeline(cls),self.pipeline(base)
                    await p._ask(self.state(p,guard),name,'system','payload','low',7)
                    await q._ask(self.state(q,guard),name,'system','payload','low',7)
                    call,reference=p.llm.chat.await_args,q.llm.chat.await_args
                    self.assertEqual(call.args,reference.args)
                    expected={**reference.kwargs,'effort':'medium' if guard==target_guard else reference.kwargs['effort']}
                    # Tutor material clamps against two independently created deadlines.
                    if name=='material' and guard is False:
                        self.assertAlmostEqual(call.kwargs['timeout'],reference.kwargs['timeout'],delta=.02)
                        expected['timeout']=call.kwargs['timeout']
                    self.assertEqual(call.kwargs,expected)

    async def test_concurrent_guard_and_tutor_do_not_mutate_client_or_global_effort(self):
        p=self.pipeline(GuardInputMedium);client=p.llm
        await asyncio.gather(p._ask(self.state(p,True),'other','s','guard','low',7),
                             p._ask(self.state(p,False),'other','s','tutor','low',7))
        self.assertCountEqual([(c.args[1],c.kwargs['effort']) for c in client.chat.await_args_list],
                             [('guard','medium'),('tutor','low')])
        self.assertIs(p.llm,client)
        from pipelines.v1.pipeline import CLASSIFY_EFFORT,GUARD_EFFORT
        self.assertEqual((CLASSIFY_EFFORT,GUARD_EFFORT),('low','low'))

    async def test_ground_medium_only_inside_c_guard_context(self):
        p=self.pipeline(GuardGroundMedium);client=p.llm
        await p._ground_response({'same':'payload'})
        original=client.chat.await_args
        token=_guard_deadline.set(time.monotonic()+28)
        try:await p._ground_response({'same':'payload'})
        finally:_guard_deadline.reset(token)
        changed=client.chat.await_args
        self.assertEqual(original.kwargs['effort'],'low')
        self.assertEqual(changed.kwargs['effort'],'medium')
        self.assertEqual(changed.kwargs['max_tokens'],original.kwargs['max_tokens'])
        self.assertEqual(changed.kwargs['timeout'],18)
        self.assertEqual(changed.args,original.args)
        self.assertIs(p.llm,client)

    async def test_safety_preserves_original_payload_budget_and_other_stages(self):
        p,q=self.pipeline(GuardSafetyMedium),self.pipeline(CheckBudgets)
        await p._fusion_safety(self.state(p,True))
        await q._fusion_safety(self.state(q,True))
        c,b=p.llm.chat.await_args,q.llm.chat.await_args
        self.assertEqual(c.args,b.args)
        self.assertEqual(c.kwargs,{**b.kwargs,'effort':'medium'})
        await p._ask(self.state(p,True),'block','s','u','low',7)
        self.assertEqual(p.llm.chat.await_args.kwargs['effort'],'low')

    async def test_source_only_promotes_tutor_selection(self):
        p,q=self.pipeline(TutorSourceMedium),self.pipeline(CharMLP50)
        for guard in (True,False):
            await p._source_response(self.state(p,guard),['A'],6)
            await q._source_response(self.state(q,guard),['A'],6)
            c,b=p.llm.chat.await_args,q.llm.chat.await_args
            self.assertEqual(c.args,b.args)
            self.assertEqual(c.kwargs,{**b.kwargs,'effort':'low' if guard else 'medium'})
        self.assertEqual(p.source_extract_timeout,8)

    async def test_answer_attribute_reaches_actual_generation_without_promoting_source(self):
        p,q=self.pipeline(TutorAnswerMedium),self.pipeline(CharMLP50)
        for obj in (p,q):
            state=self.state(obj,False);state.label=Label.ANSWER_SEEKING
            obj.llm.chat.return_value='교재 조건을 확인해 보세요.'
            await V1._generate(obj,state)
        c,b=p.llm.chat.await_args,q.llm.chat.await_args
        self.assertEqual(c.args,b.args)
        self.assertEqual(c.kwargs,{**b.kwargs,'effort':'medium'})
        await p._source_response(self.state(p,False),['A'],6)
        self.assertEqual(p.llm.chat.await_args.kwargs['effort'],'low')

    async def test_ground_integration_keeps_input_and_safety_low(self):
        p=self.pipeline(GuardGroundMedium)
        async def classify(state):
            await p._ask(state,'block','s','u','low',13)
            state.label=Label.NORMAL
        p._classify=classify
        state=await p.guard({'messages':[{'role':'user','content':'교재 질문'}],'candidate_answer':'후보 답변'})
        calls=p.llm.chat.await_args_list
        self.assertEqual(sum(c.kwargs['effort']=='medium' for c in calls),5)
        self.assertEqual(sum(c.kwargs['effort']=='low' for c in calls),2)
        self.assertEqual(state.errors,[])
        self.assertIsNone(_guard_deadline.get())
        self.assertEqual(p._slots._value,64)

    async def test_adapter_preserves_embedding_and_cancellation(self):
        client=SimpleNamespace(chat=AsyncMock(side_effect=asyncio.CancelledError),embed=AsyncMock(return_value=[[1]]),can_embed=True)
        proxy=EffortClient(client,'medium')
        self.assertTrue(proxy.can_embed)
        self.assertEqual(await proxy.embed(['text'],timeout=3),[[1]])
        client.embed.assert_awaited_once_with(['text'],timeout=3)
        with self.assertRaises(asyncio.CancelledError):await proxy.chat('s','u',effort='low',timeout=2)
        self.assertEqual(client.chat.await_args.kwargs,{'effort':'medium','timeout':2})

    def test_existing_a_c_names_classes_and_limits_preserved(self):
        self.assertIs(load('jh_cm_char_mlp50'),CharMLP50)
        self.assertIs(load('jh_np_timeouts'),CheckBudgets)
        self.assertEqual(CharMLP50.answer_effort,'low')
        for cls,base in ((GuardInputMedium,CheckBudgets),(GuardGroundMedium,CheckBudgets),(GuardSafetyMedium,CheckBudgets),
                         (TutorInputMedium,CharMLP50),(TutorSourceMedium,CharMLP50),(TutorAnswerMedium,CharMLP50)):
            for field in ('guard_deadline','deadline','guard_prepare_limit','guard_ground_timeout','guard_safety_timeout',
                          'guard_material_timeout','guard_cancel_ground_after_block','classifier_threshold','output_votes'):
                self.assertEqual(getattr(cls,field),getattr(base,field))
            for method in ('_classify','_generate','_search','_fusion_ground'):
                self.assertIs(getattr(cls,method),getattr(base,method))


if __name__=='__main__':unittest.main()
