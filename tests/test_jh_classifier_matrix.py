import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from core.schema import Label
from pipelines.kimjaehak.classifier_matrix import MatrixGate, DenseMLP
from pipelines.kimjaehak.classifier_models import CategoryModel
from pipelines.kimjaehak.peace_cancel_ground import FastCancelBlock
from pipelines.kimjaehak.voted import Pipeline as Voted
from experiments.jh_classifier_matrix import selective


class MatrixTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, kind='char_lr', category='normal', confidence=.99):
        p = MatrixGate.__new__(MatrixGate)
        p.classifier_kind = kind
        p.matrix_model = SimpleNamespace(classes=np.array([category,'off_topic']),
            probabilities=lambda texts,vectors: np.array([[confidence,1-confidence]]))
        p.classifier_embedding_model = 'text-embedding-3-small'
        p.llm = SimpleNamespace(can_embed=True,embedding_model=p.classifier_embedding_model,
            embed=AsyncMock(return_value=[[1.0]*1536]))
        return p

    def state(self,p,guard=True):
        return p._new_state({'messages':[{'role':'user','content':'교재를 설명해 주세요'}],
                             'candidate_answer':'후보와 질문은 분리'},guard)

    async def test_tutor_keeps_original_route_and_guard_uses_new_gate(self):
        p=self.pipeline(); tutor=self.state(p,False); guard=self.state(p)
        with patch.object(FastCancelBlock,'_classify',new_callable=AsyncMock) as old:
            await p._classify(tutor); old.assert_awaited_once_with(tutor)
            await p._classify(guard); self.assertEqual(old.await_count,1)
        self.assertEqual(guard.label,Label.NORMAL)
        p.llm.embed.assert_not_awaited()

    async def test_uncertain_gate_goes_to_llm_not_original_trained_gate(self):
        p=self.pipeline(confidence=.6); state=self.state(p)
        with patch.object(Voted,'_classify',new_callable=AsyncMock) as fallback, \
             patch.object(FastCancelBlock,'_classify',new_callable=AsyncMock) as old:
            await p._classify(state)
            fallback.assert_awaited_once_with(p,state); old.assert_not_awaited()

    async def test_dense_uses_last_user_only_and_embedding_errors_fall_back(self):
        p=self.pipeline('dense_mlp'); state=self.state(p)
        await p._classify(state)
        self.assertEqual(p.llm.embed.await_args.args,([state.last_user],))
        self.assertEqual(state.label,Label.NORMAL)
        for mode in ('error','mismatch'):
            p=self.pipeline('dense_mlp'); state=self.state(p)
            if mode=='error': p.llm.embed.side_effect=RuntimeError('secret value')
            else: p.llm.embedding_model='wrong-model'
            with patch.object(Voted,'_classify',new_callable=AsyncMock) as fallback:
                await p._classify(state); fallback.assert_awaited_once()
            self.assertTrue(state.errors)
            self.assertNotIn('secret value',' '.join(state.errors))

    async def test_external_cancel_propagates(self):
        p=self.pipeline('dense_mlp'); p.llm.embed.side_effect=asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError): await p._classify(self.state(p))

    async def test_harmful_subtype_confidence_still_required(self):
        p=self.pipeline(category='harmful_content')
        p.harmful_vectorizer=SimpleNamespace(transform=lambda texts: texts)
        p.harmful_classifier=SimpleNamespace(classes_=np.array(['refuse','redirect']),
            predict_proba=lambda x: np.array([[.96,.04]]))
        s=self.state(p); await p._classify(s); self.assertEqual(s.label,Label.HARMFUL_REQUEST)
        p.harmful_classifier.predict_proba=lambda x:np.array([[.6,.4]])
        with patch.object(Voted,'_classify',new_callable=AsyncMock) as fallback:
            await p._classify(self.state(p)); fallback.assert_awaited_once()

    def test_dense_validation_and_identical_normalization(self):
        a=np.ones((2,1536)); a[1]*=2
        np.testing.assert_allclose(CategoryModel._vectors(a,2)[0],CategoryModel._vectors(a,2)[1])
        for bad in (np.ones((1,5)),np.zeros((1,1536)),np.full((1,1536),np.nan)):
            with self.assertRaises(ValueError): CategoryModel._vectors(bad,1)

    def test_training_vocabulary_excludes_test_only_text(self):
        texts=['사과 학습 설명','사과 학습 설명 예제','은행 돈 관리','은행 돈 관리 예제']
        m=CategoryModel('char_lr').fit(texts,['normal','normal','off_topic','off_topic'])
        before=dict(m.vectorizer.vocabulary_)
        m.probabilities(['미관측고유문자열'])
        self.assertEqual(before,m.vectorizer.vocabulary_)
        self.assertNotIn('미관측',before)

    def test_equal_coverage_uses_ranking_and_excludes_subtypes(self):
        gold=np.array(['normal']*20,dtype=object); pred=gold.copy();pred[0]='harmful_content';pred[1]='off_topic'
        result=selective(gold,pred,np.linspace(1,.1,20))
        self.assertEqual(result['0.25']['count'],5)
        self.assertEqual(result['0.25']['category_errors'],1)

    def test_output_and_tutor_generation_are_inherited(self):
        for name in ('tutor','_generate','_fusion_safety','_fusion_ground','guard','_search'):
            self.assertIs(getattr(DenseMLP,name),getattr(FastCancelBlock,name))


if __name__=='__main__': unittest.main()
