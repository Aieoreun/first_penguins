import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import numpy as np
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50, MatrixGate
from pipelines.kimjaehak.round7_hybrid import ConcatModel, BlendModel, ConcatGate, BlendGate, FastBlendGate, CommonFallback
from tests import test_jh_classifier_matrix as matrix_tests


class HybridTests(unittest.IsolatedAsyncioTestCase):
    state=matrix_tests.MatrixTests.state

    async def test_common_fallback_bypasses_only_guard_trained_gates(self):
        from pipelines.kimjaehak.voted import Pipeline as Voted
        p=CommonFallback.__new__(CommonFallback)
        with patch.object(Voted,'_classify',new_callable=AsyncMock) as fallback, \
             patch.object(MatrixGate,'_classify',new_callable=AsyncMock) as parent:
            s=self.state(p,True);await p._classify(s)
            fallback.assert_awaited_once_with(p,s);parent.assert_not_awaited()
            self.assertTrue(s.debug['guard_unlearned_input'])
            s=self.state(p,False);await p._classify(s);parent.assert_awaited_once_with(s)
        for method in ('tutor','_generate','_fusion_ground','_fusion_safety','_search','guard'):
            self.assertIs(getattr(CommonFallback,method),getattr(CharMLP50,method))

    def test_concat_uses_dense_information_even_when_text_is_identical(self):
        texts=['동일한 학습 문장']*12;labels=['normal']*6+['off_topic']*6
        vectors=np.zeros((12,1536));vectors[:6,0]=1;vectors[6:,1]=1
        m=ConcatModel('dense_mlp').fit(texts,labels,vectors)
        p=m.probabilities([texts[0],texts[0]],vectors[[0,6]])
        self.assertEqual(m.classes[p.argmax(1)].tolist(),['normal','off_topic'])
        np.testing.assert_allclose(p,m.probabilities([texts[0],texts[0]],vectors[[0,6]]*7))
        for bad in (np.zeros((1,1536)),np.ones((1,12)),np.full((1,1536),np.nan)):
            with self.assertRaises(ValueError):m.probabilities([texts[0]],bad)

    def test_concat_also_uses_character_features_with_identical_vectors(self):
        texts=['alpha beta lesson']*6+['gamma delta lesson']*6
        m=ConcatModel('dense_mlp').fit(texts,['normal']*6+['off_topic']*6,np.ones((12,1536)))
        before=dict(m.vectorizer.vocabulary_)
        p=m.probabilities([texts[0],texts[-1]],np.ones((2,1536)))
        self.assertEqual(m.classes[p.argmax(1)].tolist(),['normal','off_topic'])
        m.probabilities(['unseen XYZ'],np.ones((1,1536)))
        self.assertEqual(before,m.vectorizer.vocabulary_)

    def test_probability_mixture_aligns_class_order(self):
        m=BlendModel()
        m.char=SimpleNamespace(classes=np.array(['normal','off_topic']),probabilities=lambda x:np.array([[.9,.1]]))
        m.dense=SimpleNamespace(classes=np.array(['off_topic','normal']),probabilities=lambda x,v:np.array([[.8,.2]]))
        m.dense_order=[1,0]
        np.testing.assert_allclose(m.probabilities(['text'],[[1]*1536]),[[.725,.275]])

    async def test_fast_gate_skips_embedding_only_for_confident_nonharmful_guard(self):
        p=FastBlendGate.__new__(FastBlendGate)
        p.matrix_model=SimpleNamespace(char=SimpleNamespace(classes=np.array(['normal','harmful_content']),
            probabilities=lambda texts:np.array([[.99,.01]])))
        p.llm=SimpleNamespace(embed=AsyncMock())
        with patch.object(MatrixGate,'_classify',new_callable=AsyncMock) as parent:
            s=self.state(p,True);await p._classify(s)
            self.assertTrue(s.debug['r7_hybrid_fast']);parent.assert_not_awaited();p.llm.embed.assert_not_awaited()
            await p._classify(self.state(p,False));self.assertEqual(parent.await_count,1)
            p.matrix_model.char.probabilities=lambda t:np.array([[.01,.99]])
            await p._classify(self.state(p,True));self.assertEqual(parent.await_count,2)
            p.matrix_model.char.probabilities=lambda t:np.array([[.7,.3]])
            await p._classify(self.state(p,True));self.assertEqual(parent.await_count,3)

    async def test_fast_gate_does_not_accept_invalid_probability(self):
        p=FastBlendGate.__new__(FastBlendGate)
        p.matrix_model=SimpleNamespace(char=SimpleNamespace(classes=np.array(['normal']),
            probabilities=lambda t:np.array([[np.nan]])))
        s=self.state(p,True)
        with patch.object(MatrixGate,'_classify',new_callable=AsyncMock) as parent:
            await p._classify(s);parent.assert_awaited_once_with(s)
        self.assertIn('hybrid_fast:ValueError',s.errors)

    async def test_hybrid_embedding_failure_and_cancellation_keep_parent_contract(self):
        for cls in (ConcatGate,BlendGate):
            p=matrix_tests.MatrixTests.pipeline(self,'dense_mlp')
            p.__class__=cls
            p.llm.embed.side_effect=asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):await p._classify(self.state(p))
            p.llm.embed.side_effect=ValueError('sensitive service detail')
            from pipelines.kimjaehak.voted import Pipeline as Voted
            with patch.object(Voted,'_classify',new_callable=AsyncMock) as fallback:
                s=self.state(p);await p._classify(s);fallback.assert_awaited_once()
            self.assertEqual(s.errors,['matrix_embedding:ValueError'])

    def test_registered_candidates_preserve_tutor_output_and_budgets(self):
        for name,cls in [('concat',ConcatGate),('blend',BlendGate),('fastblend',FastBlendGate)]:
            self.assertIs(load('jh_r7_'+name),cls)
            for method in ('tutor','_generate','_fusion_ground','_fusion_safety','_search','guard'):
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))
            for attr in ('guard_deadline','deadline','output_votes','guard_ground_timeout','classifier_embedding_timeout'):
                self.assertEqual(getattr(cls,attr),getattr(CharMLP50,attr))


if __name__=='__main__':unittest.main()
