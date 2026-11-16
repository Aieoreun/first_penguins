import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from core.state import Chunk, State
from pipelines.kimjaehak.resolved_query import ContextFast, Pipeline


class ResolvedQueryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline=Pipeline.__new__(Pipeline)
        self.pipeline.section_classifier=SimpleNamespace(predict_proba=Mock(return_value=np.array([[.6,.4]])))
        self.pipeline.section_vectorizer=SimpleNamespace(transform=Mock())
        self.pipeline.searcher=SimpleNamespace(search=Mock(side_effect=lambda text,**k:[Chunk('old' if '\n' in text else 'new','text')]))
        self.pipeline.llm=SimpleNamespace(chat=AsyncMock(return_value='{"query":"현재 질문의 구체적 대상"}'))
        self.state=State(messages=[{'role':'user','content':'이전 질문'}, {'role':'assistant','content':'이전 내용'},
                                   {'role':'user','content':'현재 질문'}])

    async def search(self,pipeline,state,extra=''):
        state.chunks=[Chunk('new' if len(state.messages)==1 else 'old','found')]

    async def run_search(self):
        async def search(p,s,extra=''):await self.search(p,s,extra)
        with patch.object(ContextFast,'_search',search):await self.pipeline._search(self.state)

    async def test_only_search_query_changes_and_conversation_is_preserved(self):
        old=copy.deepcopy(self.state.messages)
        await self.run_search()
        self.assertEqual(self.state.messages,old)
        self.assertEqual(self.state.chunks[0].section_id,'new')
        self.assertIsNone(self.state.label)

    async def test_invalid_query_or_api_failure_restores_original_search(self):
        for raw in ('{}','{"query":"PY-01-01"}','{"query":""}'):
            self.pipeline.llm.chat.return_value=raw
            await self.run_search()
            self.assertEqual(self.state.chunks[0].section_id,'old')
        self.pipeline.llm.chat.side_effect=TimeoutError
        await self.run_search()
        self.assertEqual(self.state.chunks[0].section_id,'old')

    async def test_guard_never_runs_query_resolver(self):
        self.state._query_guard_endpoint=True
        await self.run_search()
        self.pipeline.llm.chat.assert_not_awaited()
        self.pipeline.section_classifier.predict_proba.assert_not_called()
        self.assertEqual(self.state.chunks[0].section_id,'old')

    async def test_confident_section_or_single_turn_skips_resolver(self):
        self.pipeline.section_classifier.predict_proba.return_value=np.array([[.999,.001]])
        await self.run_search()
        self.pipeline.llm.chat.assert_not_awaited()
        self.state.messages=self.state.messages[-1:]
        await self.run_search()
        self.pipeline.llm.chat.assert_not_awaited()

    async def test_parent_cancellation_cleans_up_backup(self):
        started,closed=asyncio.Event(),asyncio.Event()
        async def search(p,s,extra=''):
            started.set()
            try:await asyncio.Future()
            finally:closed.set()
        async def chat(*a,**k):await asyncio.Future()
        self.pipeline.llm.chat.side_effect=chat
        with patch.object(ContextFast,'_search',search):
            task=asyncio.create_task(self.pipeline._search(self.state))
            await asyncio.wait_for(started.wait(),1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(closed.is_set())

    def test_loader_and_guard_classification_output_unchanged(self):
        from pipelines import load
        self.assertIs(load('jh_r3_resolved_query'),Pipeline)
        for method in ('guard','_classify','_check_output','_generate'):
            self.assertIs(getattr(Pipeline,method),getattr(ContextFast,method))

    async def test_minimal_variant_changes_only_resolver_effort(self):
        from pipelines import load
        minimal = load('jh_r3_resolved_minimal')
        self.pipeline.__class__ = minimal
        await self.run_search()
        self.assertEqual(self.pipeline.llm.chat.call_args.kwargs['effort'], 'minimal')
        self.assertEqual(self.pipeline.llm.chat.call_args.kwargs['timeout'], 4)
        for method in ('guard', '_classify', '_check_output', '_generate'):
            self.assertIs(getattr(minimal, method), getattr(ContextFast, method))

    async def test_topic_selector_preserves_original_text_and_default_context(self):
        from pipelines import load
        self.pipeline.__class__ = load('jh_r3_topic_select')
        self.pipeline.llm.chat.return_value = '{"choice":"full"}'
        await self.run_search()
        self.assertEqual(self.state.chunks[0].section_id, 'old')
        self.assertEqual(self.state.errors, [])
        self.pipeline.llm.chat.return_value = '{"choice":"last"}'
        self.assertEqual(await self.pipeline._resolve_query(self.state), self.state.last_user)
        await self.run_search()
        self.assertEqual(self.state.chunks[0].section_id, 'new')
        self.pipeline.llm.chat.return_value = '{"choice":"rewrite"}'
        await self.run_search()
        self.assertEqual(self.state.chunks[0].section_id, 'old')
        self.assertIn('query_resolve:ValueError', self.state.errors)
