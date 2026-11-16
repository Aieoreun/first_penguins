import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.source_extract import GatedSection, MaterialSection, SelectedSection


class SourceExtractTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self, raw='{"section":"last"}'):
        p=SelectedSection.__new__(SelectedSection)
        p.searcher=SimpleNamespace(search=lambda *a,**kw:[Chunk('last','last source')],section_text=lambda sid:sid+' source material')
        p.llm=SimpleNamespace(chat=AsyncMock(return_value=raw))
        s=State(messages=[{'role':'user','content':'current question'}],label=Label.NORMAL,section='old',chunks=[Chunk('old','old source'),Chunk('third','third source')])
        s._source_extract_deadline=time.monotonic()+4
        return p,s

    async def test_one_source_call_then_exact_excerpt(self):
        p,s=self.setup_case()
        await p._generate(s)
        p.llm.chat.assert_awaited_once()
        self.assertEqual(s.section,'last')
        self.assertEqual(s.answer,'관련 교재 내용을 인용할게요.\n\nlast source material')
        payload=json.loads(p.llm.chat.call_args.args[1])
        self.assertEqual([m['section'] for m in payload['materials']],['old','last','third'])
        self.assertLess(p.llm.chat.call_args.kwargs['timeout'],4)

    async def test_invalid_source_uses_original_excerpt_without_retry(self):
        p,s=self.setup_case('{"section":"outside"}')
        await p._generate(s)
        self.assertEqual(s.section,'old')
        self.assertEqual(s.errors,['source_extract:ValueError'])
        self.assertEqual(s.citations,['old'])
        self.assertEqual(p.llm.chat.await_count,1)

    async def test_known_section_or_exhausted_time_skips_selector(self):
        for debug in [{'learned_section':True},{'tutor_section_gate':.91},{}]:
            p,s=self.setup_case();s.debug=debug
            if not debug:s._source_extract_deadline=time.monotonic()
            await p._generate(s)
            p.llm.chat.assert_not_awaited()
            self.assertEqual(s.citations,['old'])

    async def test_cancelled_selection_is_not_retried(self):
        p,s=self.setup_case();p.llm.chat.side_effect=asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):await p._generate(s)
        self.assertEqual(p.llm.chat.await_count,1)

    async def test_policy_and_guard_bypass_selector(self):
        for guard,label in [(True,Label.NORMAL),(False,Label.ANSWER_SEEKING)]:
            p,s=self.setup_case();s._extract_guard=guard;s.label=label
            with patch.object(ContextFast,'_generate',new_callable=AsyncMock) as original:
                await p._generate(s)
                original.assert_awaited_once_with(s)
            p.llm.chat.assert_not_awaited()

    def test_all_compositions_preserve_guard_markers(self):
        from pipelines import load
        for name,cls in [('jh_r3_material_extract',MaterialSection),('jh_r3_gated_extract',GatedSection),('jh_r3_selected_extract',SelectedSection)]:
            self.assertIs(load(name),cls)
            p=cls.__new__(cls);s=p._new_state({'messages':[]},True)
            self.assertTrue(s._extract_guard)
            marker='_material_source_guard' if cls is MaterialSection else '_section_gate_guard'
            self.assertTrue(getattr(s,marker))
            for method in ('guard','_check_output'):
                self.assertIs(getattr(cls,method),getattr(ContextFast,method))
