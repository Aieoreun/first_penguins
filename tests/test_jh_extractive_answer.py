import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.kb import load_chunks
from core.schema import ANSWER_MAX_CHARS, Label
from core.search import Searcher
from core.state import State
from pipelines.kimjaehak.extractive_answer import Chunk, ContextFast, Section


class ExtractionTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self, cls=Section):
        p = cls.__new__(cls)
        p.searcher = SimpleNamespace(section_text=lambda sid: '# S Source\nAllowed source only')
        p.llm = SimpleNamespace(chat=AsyncMock())
        s = State(messages=[{'role':'user','content':'question, private user detail'}], label=Label.NORMAL, section='S')
        return p, s

    async def test_only_selected_source_is_returned_without_llm_or_user_data(self):
        p,s = self.setup_case()
        await p._generate(s)
        self.assertEqual(s.citations, ['S'])
        self.assertEqual(s.answer, '관련 교재 내용을 인용할게요.\n\n# S Source\nAllowed source only')
        self.assertNotIn('private', s.answer)
        p.llm.chat.assert_not_awaited()

    async def test_policy_and_guard_paths_are_unchanged(self):
        for label, guard in [(Label.ANSWER_SEEKING,False),(Label.HARMFUL_REQUEST,False),(Label.NORMAL,True)]:
            p,s = self.setup_case()
            s.label, s._extract_guard = label, guard
            with patch.object(ContextFast, '_generate', new_callable=AsyncMock) as original:
                await p._generate(s)
                original.assert_awaited_once_with(s)

    async def test_missing_or_oversize_source_falls_back_without_truncating(self):
        for text in ['', 'a' * ANSWER_MAX_CHARS]:
            p,s = self.setup_case()
            p.searcher.section_text = lambda sid: text
            with patch.object(ContextFast, '_generate', new_callable=AsyncMock) as original:
                await p._generate(s)
                original.assert_awaited_once_with(s)

    async def test_chunk_ranking_cannot_change_selected_section(self):
        p,s = self.setup_case(Chunk)
        p.searcher.chunks = [SimpleNamespace(section_id='other',text='Wrong'),SimpleNamespace(section_id='S',text='Low'),SimpleNamespace(section_id='S',text='Best')]
        p.searcher.bm25 = SimpleNamespace(scores=lambda query: [100,1,2])
        await p._generate(s)
        self.assertEqual(s.answer, '관련 교재 내용을 인용할게요.\n\nBest')
        self.assertEqual(s.section, 'S')

    def test_bundled_sections_fit_response_limit_and_guard_helpers_match(self):
        from pipelines import load
        self.assertIs(load('jh_r3_extract_section'),Section)
        self.assertIs(load('jh_r3_extract_chunk'),Chunk)
        kb = Searcher(load_chunks(Path(__file__).resolve().parents[1] / 'kb'))
        for sid in kb.section_ids():
            self.assertLessEqual(len('관련 교재 내용을 인용할게요.\n\n' + kb.section_text(sid)), ANSWER_MAX_CHARS)
        for cls in (Section,Chunk):
            for method in ('guard','_classify','_search','_check_output'):
                self.assertIs(getattr(cls,method),getattr(ContextFast,method))
