import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
import numpy as np
from core.schema import Label
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round8_failure_fallback import FailureInput,FailureGround,FailureBoth,FailureBlock,TutorFailureInput,FailureGroundDeadline,FailureDeadlineBoth,ResilientGuard,ResilientBoth,input_checks_failed
from core.state import State

class FailureTests(unittest.IsolatedAsyncioTestCase):
    def state(self):
        return SimpleNamespace(_material_budget_guard=True,label=Label.NORMAL,debug={'valid_block_votes':0,'valid_other_votes':0,'self_info_votes':[],'out_of_material':False},errors=['block:TimeoutError','other:TimeoutError','material:TimeoutError'],last_user='학습 질문',candidate_answer='근거 없는 문장',grounded=True,unsupported=[])

    async def test_complete_failure_uses_local_decision_without_another_call(self):
        p=FailureInput.__new__(FailureInput);p.matrix_model=SimpleNamespace(classes=np.array(['normal','jailbreak_injection']),probabilities=Mock(return_value=[[.3,.7]]));s=self.state()
        with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
        self.assertEqual(s.label,Label.JAILBREAK_INJECTION);self.assertTrue(s.debug['r8_failure_input'])
        self.assertEqual(len(s.errors),3)

    async def test_partial_success_tutor_and_external_cancellation_preserved(self):
        for mode in ['block_vote','other_vote','material','tutor','label']:
            p=FailureInput.__new__(FailureInput);p.matrix_model=Mock();s=self.state()
            if mode=='block_vote':s.debug['valid_block_votes']=1
            if mode=='other_vote':s.debug['valid_other_votes']=1
            if mode=='material':s.errors.remove('material:TimeoutError')
            if mode=='tutor':s._material_budget_guard=False
            if mode=='label':s.label=Label.OUT_OF_MATERIAL
            with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
            p.matrix_model.probabilities.assert_not_called()
        with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock,side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):await p._classify(self.state())

    async def test_ground_fallback_never_replaces_completed_vote_or_cancellation(self):
        for count,expected in [(0,True),(1,False)]:
            p=FailureGround.__new__(FailureGround);p.span_tagger=SimpleNamespace(predict=Mock(return_value=(['없는 문장'],.99)));s=self.state();s.debug['fusion_completed_ground_votes']=count
            with patch.object(BinaryGuard,'_fusion_ground',new_callable=AsyncMock):await p._fusion_ground(s)
            self.assertEqual(s.debug.get('r8_failure_ground',False),expected)
            self.assertEqual(s.unsupported,['없는 문장'] if expected else [])
        with patch.object(BinaryGuard,'_fusion_ground',new_callable=AsyncMock,side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):await p._fusion_ground(self.state())

    def test_combined_paths(self):
        self.assertIs(FailureBoth._classify,FailureInput._classify)
        self.assertIs(FailureBoth._fusion_ground,FailureGround._fusion_ground)
        self.assertIs(FailureBoth.tutor,BinaryGuard.tutor)
        self.assertEqual(FailureBoth.guard_deadline,28)

    async def test_partial_block_failure_preserves_votes_and_tutor(self):
        for mode in ('failed','valid_none','blocked','healthy','tutor','invalid'):
            p=FailureBlock.__new__(FailureBlock)
            p.matrix_model=SimpleNamespace(classes=np.array(['normal','jailbreak_injection','off_topic']),
                probabilities=Mock(return_value=[[.2,.5,.3]]))
            s=State(messages=[{'role':'user','content':'검사 질문'}],label=Label.OUT_OF_MATERIAL)
            s._material_budget_guard=True;s.debug={'valid_block_votes':0,'valid_other_votes':2}
            s.errors=['block:TimeoutError']
            if mode=='valid_none':s.debug['valid_block_votes']=1
            if mode=='blocked':s.label=Label.OFF_TOPIC
            if mode=='healthy':s.errors=[]
            if mode=='tutor':s._material_budget_guard=False
            if mode=='invalid':p.matrix_model.probabilities.return_value=[[float('nan'),.5,.3]]
            before=s.label
            with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
            self.assertEqual(s.label,Label.JAILBREAK_INJECTION if mode=='failed' else before)
            self.assertEqual(bool(s.debug.get('r8_failure_block')),mode=='failed')
        with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock,side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):await p._classify(s)
        self.assertIs(load('jh_r8_failure_block'),FailureBlock)
        self.assertIs(FailureBlock.tutor,BinaryGuard.tutor)

    async def test_tutor_failure_preserves_guard_and_harmful_subtype(self):
        p=TutorFailureInput.__new__(TutorFailureInput)
        p.matrix_model=SimpleNamespace(classes=np.array(['normal','harmful_content']),probabilities=Mock(return_value=[[.1,.9]]))
        p.harmful_vectorizer=SimpleNamespace(transform=Mock(return_value=[[1]]))
        p.harmful_classifier=SimpleNamespace(classes_=np.array(['redirect','refuse']),predict_proba=Mock(return_value=[np.array([.99,.01])]))
        p.learned_harmful_threshold=.9
        s=self.state()
        with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
        p.matrix_model.probabilities.assert_not_called()
        s._material_budget_guard=False
        with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
        self.assertEqual(s.label,Label.HARMFUL_SPEECH)
        self.assertTrue(s.debug['r8_tutor_failure_input'])
        s.label=Label.NORMAL;s.debug['valid_other_votes']=1;p.matrix_model.probabilities.reset_mock()
        with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
        p.matrix_model.probabilities.assert_not_called()
        self.assertIs(load('jh_r8_tutor_failure'),TutorFailureInput)

    async def test_ground_deadline_recovers_only_when_no_vote_survived(self):
        p=FailureGroundDeadline.__new__(FailureGroundDeadline)
        p.span_tagger=SimpleNamespace(predict=Mock(return_value=(['없는 문장'],.99)))
        for votes in [0,1]:
            s=self.state()
            if votes:s.unsupported=['근거 없는 문장']
            async def slow():
                try:await asyncio.Future()
                finally:s.debug['fusion_completed_ground_votes']=votes
            self.assertFalse(await p._fusion_stage(slow(),.01,s,'ground'))
            self.assertEqual(s.unsupported,['없는 문장'] if votes==0 else ['근거 없는 문장'])
            self.assertIn('ground:Deadline',s.errors)
        p.span_tagger.predict.reset_mock();s=self.state();started=asyncio.Event()
        async def slow_cancel():started.set();await asyncio.Future()
        task=asyncio.create_task(p._fusion_stage(slow_cancel(),1,s,'ground'))
        await started.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        p.span_tagger.predict.assert_not_called()
        self.assertIs(load('jh_r8_failure_deadline_both'),FailureDeadlineBoth)
        self.assertEqual(FailureDeadlineBoth.guard_deadline,28)

    async def test_combined_failure_paths_and_healthy_noop(self):
        for cls,guard,expected in [(ResilientGuard,True,True),(ResilientGuard,False,False),(ResilientBoth,True,True),(ResilientBoth,False,True)]:
            p=cls.__new__(cls)
            p.matrix_model=SimpleNamespace(classes=np.array(['normal','jailbreak_injection']),probabilities=Mock(return_value=[[.3,.7]]))
            s=State(messages=[{'role':'user','content':'검사 질문'}],label=Label.NORMAL)
            s._material_budget_guard=guard;s.debug={'valid_block_votes':0,'valid_other_votes':0,'self_info_votes':[],'out_of_material':False}
            s.errors=['block:TimeoutError','other:TimeoutError','material:TimeoutError']
            with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
            self.assertEqual(s.label,Label.JAILBREAK_INJECTION if expected else Label.NORMAL)
            s.label=Label.NORMAL;s.errors=[];s.debug={'valid_block_votes':2,'valid_other_votes':2};p.matrix_model.probabilities.reset_mock()
            with patch.object(BinaryGuard,'_classify',new_callable=AsyncMock):await p._classify(s)
            p.matrix_model.probabilities.assert_not_called()
        self.assertIs(load('jh_r8_resilient_guard'),ResilientGuard)
        self.assertIs(load('jh_r8_resilient_both'),ResilientBoth)

if __name__=='__main__':unittest.main()
