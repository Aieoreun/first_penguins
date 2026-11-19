import asyncio
import unittest
from unittest.mock import patch
from pipelines import load
load('jh_cm_char_mlp50')
from core.schema import Label
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.context_output import Pipeline as ContextOutput
from pipelines.kimjaehak.round7_adaptive_ground import GroundThree,AdaptiveGround,AdaptiveThree
from tests import test_jh_reasoning_ablation as reasoning


class AdaptiveTests(unittest.IsolatedAsyncioTestCase):
    pipeline=reasoning.ReasoningTests.pipeline
    state=reasoning.ReasoningTests.state

    def ready(self,cls=AdaptiveGround):
        p=self.pipeline(cls);s=self.state(p,True)
        s.label=Label.NORMAL;s.candidate_answer='앞부분 bad 뒤부분'
        return p,s

    async def test_agreement_ends_after_two_original_votes(self):
        for vote in ([],['bad']):
            p,s=self.ready();calls=[]
            async def check(instance,trial):
                calls.append(trial);trial.unsupported=list(vote)
            with patch.object(ContextOutput,'_check_output',new=check):await p._fusion_ground(s)
            self.assertEqual(len(calls),2)
            self.assertEqual(s.unsupported,vote)
            self.assertEqual(s.grounded,not vote)
            self.assertTrue(s.debug['r7_ground_early'])
            self.assertEqual(s.debug['fusion_completed_ground_votes'],2)

    async def test_disagreement_and_error_launch_three_more_and_use_same_consensus(self):
        for first_error in (False,True):
            p,s=self.ready();calls=[]
            async def check(instance,trial):
                i=len(calls);calls.append(trial)
                if first_error and i==0:trial.errors.append('ground:parse')
                trial.unsupported=[] if i==1 else ['bad']
            with patch.object(ContextOutput,'_check_output',new=check):await p._fusion_ground(s)
            self.assertEqual(len(calls),5)
            self.assertEqual(s.unsupported,['bad'])
            self.assertEqual(s.debug['fusion_completed_ground_votes'],4 if first_error else 5)
            self.assertNotIn('r7_ground_early',s.debug)

    async def test_cancellation_preserves_completed_votes_in_either_batch(self):
        for second_batch in (False,True):
            p,s=self.ready();calls=[];cancelled=[];ready=asyncio.Event()
            async def check(instance,trial):
                i=len(calls);calls.append(trial)
                if i==0 or (second_batch and i==2):
                    trial.unsupported=['bad'];return
                if second_batch and i==1:
                    trial.unsupported=[];return
                if (second_batch and i==4) or (not second_batch and i==1):ready.set()
                try:await asyncio.Event().wait()
                finally:cancelled.append(i)
            with patch.object(ContextOutput,'_check_output',new=check):
                task=asyncio.create_task(p._fusion_ground(s))
                await asyncio.wait_for(ready.wait(),1);await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
            self.assertEqual(s.unsupported,['bad'])
            self.assertEqual(s.debug['fusion_completed_ground_votes'],3 if second_batch else 1)
            self.assertEqual(len(calls),5 if second_batch else 2)
            self.assertEqual(sorted(cancelled),[3,4] if second_batch else [1])

    async def test_failed_votes_are_excluded_and_never_create_false_span(self):
        p,s=self.ready();calls=[]
        async def check(instance,trial):
            calls.append(trial);trial.unsupported=['bad'];raise ValueError('bad result')
        with patch.object(ContextOutput,'_check_output',new=check):await p._fusion_ground(s)
        self.assertEqual(len(calls),5)
        self.assertEqual(s.unsupported,[])
        self.assertEqual(s.debug['fusion_completed_ground_votes'],0)
        self.assertEqual(len(s.errors),5)

    async def test_three_initial_votes_require_all_three_to_agree(self):
        for dissent in (False,True):
            p,s=self.ready(AdaptiveThree);calls=[]
            async def check(instance,trial):
                index=len(calls);calls.append(trial)
                trial.unsupported=[] if dissent and index==2 else ['bad']
            with patch.object(ContextOutput,'_check_output',new=check):await p._fusion_ground(s)
            self.assertEqual(len(calls),5 if dissent else 3)
            self.assertEqual(s.unsupported,['bad'])
            self.assertEqual(s.debug.get('r7_ground_early',False),not dissent)
        self.assertIs(load('jh_r7_adaptive3'),AdaptiveThree)
        self.assertEqual(AdaptiveThree.guard_ground_timeout,CharMLP50.guard_ground_timeout)

    async def test_three_parallel_and_original_five_remain_separate(self):
        for cls,count in ((GroundThree,3),(CharMLP50,5)):
            p,s=self.ready(cls);calls=[]
            async def check(instance,trial):calls.append(trial);trial.unsupported=[]
            with patch.object(ContextOutput,'_check_output',new=check):await p._fusion_ground(s)
            self.assertEqual(len(calls),count)
        for name,cls in [('jh_r7_ground3',GroundThree),('jh_r7_ground_adaptive',AdaptiveGround)]:
            self.assertIs(load(name),cls)
            for method in ('tutor','_generate','_classify','_search','guard','_fusion_safety'):
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))
            self.assertEqual(cls.guard_ground_timeout,CharMLP50.guard_ground_timeout)


if __name__=='__main__':unittest.main()
