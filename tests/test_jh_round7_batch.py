import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
import numpy as np
from pipelines import load
load('jh_r7_batchblend')
from pipelines.kimjaehak.round7_batch import BatchedBlend, BatchedConcat, BatchProxy
from pipelines.kimjaehak.round7_hybrid import BlendGate, ConcatGate


class BatchTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self):
        p=BatchedBlend.__new__(BatchedBlend)
        self.search_vectors=[];self.classifier_vectors=[]
        def search(query,vector=None):
            self.search_vectors.append(vector)
            return [SimpleNamespace(section_id='unit',text='unit\n학습 자료')]
        p.searcher=SimpleNamespace(has_embeddings=True,search=search)
        p.classifier_embedding_model='text-embedding-3-small'
        async def embed(texts,*args,**kwargs):
            await asyncio.sleep(0)
            return [[float(len(t))]*1536 for t in texts]
        p.llm=SimpleNamespace(can_embed=True,embedding_model='openai/text-embedding-3-small',embed=AsyncMock(side_effect=embed))
        def probabilities(texts,vectors):
            self.classifier_vectors.append(vectors[0])
            return np.array([[.99,.01]])
        p.matrix_model=SimpleNamespace(classes=np.array(['normal','off_topic']),probabilities=probabilities)
        return p

    def state(self,p,text='교재의 조건을 설명해 줘',guard=True):
        return p._new_state({'messages':[{'role':'user','content':text}]},guard)

    async def test_batch_separates_retrieval_and_classifier_text_and_reuses_vector(self):
        p=self.pipeline();s=self.state(p);original_llm=p.llm
        await p._search(s,'긴 후보 답변을 검색에만 포함')
        self.assertEqual(p.llm.embed.await_count,1)
        texts=p.llm.embed.await_args.args[0]
        self.assertEqual(len(texts),2);self.assertEqual(texts[1],s.last_user)
        self.assertNotEqual(texts[0],texts[1])
        await p._classify(s)
        self.assertEqual(p.llm.embed.await_count,1)
        self.assertEqual(self.search_vectors[0][0],float(len(texts[0])))
        self.assertEqual(self.classifier_vectors[0][0],float(len(texts[1])))
        self.assertTrue(s.debug['r7_embedding_reused'])
        self.assertFalse(hasattr(s,'_r7_classifier_vector'))
        self.assertIs(p.llm,original_llm)

    async def test_identical_query_embeds_once_and_concurrent_requests_are_isolated(self):
        p=self.pipeline();a=self.state(p);b=self.state(p,'다른 질문의 조건과 계산 기준을 자세히 알려줘')
        async def run(s):await p._search(s);await p._classify(s)
        await asyncio.gather(run(a),run(b))
        self.assertEqual(p.llm.embed.await_count,2)
        self.assertTrue(all(len(c.args[0])==1 for c in p.llm.embed.await_args_list))
        self.assertEqual(sorted(v[0] for v in self.classifier_vectors),sorted([float(len(a.last_user)),float(len(b.last_user))]))

    async def test_tutor_and_model_mismatch_do_not_batch(self):
        p=self.pipeline();s=self.state(p,guard=False)
        await p._search(s,'후보')
        self.assertEqual(len(p.llm.embed.await_args.args[0]),1)
        self.assertFalse(hasattr(s,'_r7_classifier_vector'))
        p.llm.embedding_model='different-model';s=self.state(p)
        await p._search(s,'후보')
        self.assertEqual(len(p.llm.embed.await_args.args[0]),1)
        self.assertFalse(hasattr(s,'_r7_classifier_vector'))

    async def test_invalid_batch_is_not_cached_and_cancel_propagates(self):
        p=self.pipeline();s=self.state(p)
        p.llm.embed.side_effect=None;p.llm.embed.return_value=[[1]*1536]
        with self.assertRaises(ValueError):await BatchProxy(p.llm,s).embed(['검색 질의'],timeout=3)
        self.assertFalse(hasattr(s,'_r7_classifier_vector'))
        p.llm.embed.side_effect=asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):await BatchProxy(p.llm,s).embed(['검색 질의'])

    async def test_failed_search_batch_retries_classification_without_wrong_cache(self):
        p=self.pipeline();s=self.state(p)
        p.llm.embed.side_effect=[ValueError('private detail'),[[1]*1536]]
        await p._search(s,'후보')
        self.assertFalse(hasattr(s,'_r7_classifier_vector'))
        await p._classify(s)
        self.assertEqual(p.llm.embed.await_count,2)
        self.assertTrue(s.debug['matrix_gate'])
        self.assertEqual(s.errors,['embed:ValueError'])

    def test_registry_and_parent_budgets_preserved(self):
        for name,cls,parent in [('batchconcat',BatchedConcat,ConcatGate),('batchblend',BatchedBlend,BlendGate)]:
            self.assertIs(load('jh_r7_'+name),cls)
            for attr in ('guard_deadline','deadline','classifier_threshold','classifier_embedding_timeout'):
                self.assertEqual(getattr(cls,attr),getattr(parent,attr))
            for method in ('tutor','_generate','_fusion_ground','_fusion_safety'):
                self.assertIs(getattr(cls,method),getattr(parent,method))


if __name__=='__main__':unittest.main()
