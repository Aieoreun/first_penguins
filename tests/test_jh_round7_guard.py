import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
import numpy as np
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_guard import BoundaryModel, BoundaryGate, VerdictFirst, verdict_first_index
from pipelines.kimjaehak.voted_output import consensus_index
from tests import test_jh_reasoning_ablation as reasoning


class GuardTests(unittest.IsolatedAsyncioTestCase):
    pipeline=reasoning.ReasoningTests.pipeline
    state=reasoning.ReasoningTests.state

    def test_majority_verdict_precedes_span_medoid_but_keeps_original_vote(self):
        candidate='alpha beta gamma'
        votes=[[],[],['alpha'],['beta'],['gamma']]
        self.assertEqual(votes[consensus_index(candidate,votes)],[])
        self.assertEqual(votes[verdict_first_index(candidate,votes)],['alpha'])
        for votes in ([[],[],['alpha']], [[],['alpha']], [['alpha'],['alpha beta'],['beta']]):
            i=verdict_first_index(candidate,votes)
            self.assertIn(votes[i],votes)
        self.assertEqual(verdict_first_index(candidate,[[],['alpha']]),consensus_index(candidate,[[],['alpha']]))

    async def test_cancel_drains_pending_votes_and_preserves_completed_majority(self):
        p=self.pipeline(VerdictFirst);s=self.state(p,True)
        s.candidate_answer='alpha beta gamma'
        ready,drained=asyncio.Event(),asyncio.Event();i=0
        async def chat(*args,**kwargs):
            nonlocal i
            n=i;i+=1
            if n==4:
                ready.set()
                try:await asyncio.Event().wait()
                finally:drained.set()
            return json.dumps({'unsupported':[[],['alpha'],['beta'],['gamma']][n]})
        p.llm.chat.side_effect=chat
        task=asyncio.create_task(p._fusion_ground(s))
        await asyncio.wait_for(ready.wait(),1);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(drained.is_set())
        self.assertFalse(s.grounded)
        self.assertEqual(s.unsupported,['alpha'])
        self.assertEqual(s.debug['fusion_completed_ground_votes'],4)
        self.assertEqual(p.llm.chat.await_count,5)

    async def test_failed_votes_are_excluded_and_refinement_remains_single(self):
        p=self.pipeline(VerdictFirst);s=self.state(p,True);s.candidate_answer='alpha beta gamma'
        p.llm.chat.side_effect=[RuntimeError(),'{"unsupported":["alpha"]}',
            '{"unsupported":["beta"]}','{"unsupported":["gamma"]}','{"unsupported":[]}']
        calls=[]
        def predict(candidate):calls.append(candidate);return ['alpha beta'],.99
        p.span_tagger=SimpleNamespace(predict=predict)
        await p._fusion_ground(s)
        self.assertEqual(s.unsupported,['alpha beta'])
        self.assertEqual(calls,[s.candidate_answer])
        self.assertEqual(s.debug['fusion_completed_ground_votes'],4)
        self.assertTrue(any('RuntimeError' in e for e in s.errors))
        self.assertTrue(all(c.kwargs['effort']=='low' and c.kwargs['max_tokens']==1500 for c in p.llm.chat.await_args_list))

    def test_boundary_model_fits_train_only_and_accepts_unseen_spacing(self):
        texts=['alpha beta lesson']*4+['gamma delta policy']*4
        labels=['normal']*4+['off_topic']*4
        m=BoundaryModel('char_mlp').fit(texts,labels)
        p=m.probabilities(['alpha   beta lesson','gamma delta policy'])
        self.assertEqual(p.shape,(2,2));self.assertTrue(np.allclose(p.sum(axis=1),1))
        self.assertEqual(m.classes[p.argmax(axis=1)].tolist(),['normal','off_topic'])
        self.assertNotIn('a b',m.vectorizer.vocabulary_)

    def test_tutor_and_unchanged_guard_stages_preserved(self):
        for name,cls in (('jh_r7_boundary',BoundaryGate),('jh_r7_verdict',VerdictFirst)):
            self.assertIs(load(name),cls)
            for method in ('tutor','_generate','_classify','_fusion_safety','_search'):
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))
            for attr in ('guard_deadline','deadline','output_votes','guard_ground_timeout','guard_cancel_ground_after_block'):
                self.assertEqual(getattr(cls,attr),getattr(CharMLP50,attr))


if __name__=='__main__':unittest.main()
