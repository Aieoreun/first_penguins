import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from copy import copy

from pipelines import load
Kiwi = load('jh_r6_kiwi_bigram')
Resolved = load('jh_r6_resolved_focused')
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_retrieval import MorphBM25, content_tokens
from core.search import BM25, Searcher, tokens
from core.state import Chunk


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    def test_formula_same_when_tokenization_same(self):
        texts=['질문 원금 금리 123 abc','질문 코드 def 함수','내용']
        original=BM25(texts); changed=MorphBM25(texts,tokens)
        for query in ('질문 금리 원금','abc 123 def','없는내용',''):
            self.assertEqual(original.scores(query),changed.scores(query))

    def test_content_pairs_and_identifiers_survive(self):
        kiwi=SimpleNamespace(tokenize=lambda text:[SimpleNamespace(form=x,tag=t) for x,t in
            [('간편','NNG'),('결제','NNG'),('는','JX'),('3.5','SN'),('fruit_basket','SL')]])
        result=content_tokens('간편 결제는 3.5 fruit_basket',kiwi)
        self.assertIn('간편_결제',result)
        self.assertIn('3.5',result)
        self.assertIn('fruit_basket',result)

    async def test_search_view_is_request_local_and_guard_stays_original(self):
        p=Kiwi.__new__(Kiwi)
        p.searcher=Searcher([Chunk('A','alpha'),Chunk('B','beta')])
        p._tutor_searcher=copy(p.searcher)
        p._tutor_searcher.bm25=SimpleNamespace(scores=lambda q:[0,1])
        p.llm=SimpleNamespace(can_embed=False)
        tutor=p._new_state({'messages':[{'role':'user','content':'alpha'}]},False)
        guard=p._new_state({'messages':[{'role':'user','content':'alpha'}]},True)
        await asyncio.gather(p._search(tutor),p._search(guard))
        self.assertEqual(tutor.chunks[0].section_id,'B')
        self.assertEqual(guard.chunks[0].section_id,'A')
        self.assertIsInstance(p.searcher.bm25,BM25)
        self.assertEqual(tutor.messages,guard.messages)

    def test_resolver_mro_preserves_material_and_source_budgets(self):
        p=Resolved.__new__(Resolved)
        s=p._new_state({'messages':[{'role':'user','content':'question'}]},False)
        self.assertFalse(s._material_budget_guard)
        self.assertFalse(s._query_guard_endpoint)
        self.assertEqual((p.query_resolve_effort,p.query_resolve_timeout),('minimal',4))
        self.assertEqual((p.tutor_material_timeout,p.tutor_prepare_limit,p.deadline,p.source_extract_timeout),(13,16,26,8))
        self.assertIs(Resolved._source_response,Base._source_response)
        self.assertIs(Resolved._generate,Base._generate)
        self.assertIs(Resolved.guard,Base.guard)
        self.assertIs(Resolved._check_output,Base._check_output)

    async def test_resolved_search_keeps_actual_conversation_and_endpoint_flags(self):
        p=Resolved.__new__(Resolved)
        s=p._new_state({'messages':[{'role':'user','content':'prior'},
            {'role':'assistant','content':'reply'},{'role':'user','content':'current question'}]},False)
        before=list(s.messages)
        p._should_resolve=lambda state:True
        p._resolve_query=AsyncMock(return_value='resolved question')
        async def search(state,extra=''):
            await asyncio.sleep(0)
            state.chunks=[Chunk('A',state.messages[-1]['content'])]
        with patch.object(Base,'_search',side_effect=search):
            await p._search(s)
        self.assertEqual(s.chunks[0].text,'resolved question')
        self.assertEqual(s.messages,before)
        self.assertFalse(s._material_budget_guard)
        self.assertFalse(s._query_guard_endpoint)


if __name__=='__main__':unittest.main()
