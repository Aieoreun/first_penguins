from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from core.kb import load_chunks
from core.schema import ANSWER_MAX_CHARS, Label
from core.search import Searcher
from core.state import Chunk, State
from pipelines.kimjaehak.extractive_answer import Section
from pipelines.kimjaehak.focused_excerpt import Grounded, Pipeline, Quoted, Selected
from pipelines.kimjaehak.source_extract import GroundedSelected, QuoteSelected, SelectedSection


class FocusedExcerptTests(unittest.IsolatedAsyncioTestCase):
    async def test_relevance_changes_order_without_dropping_source_or_calling_llm(self):
        p = Pipeline.__new__(Pipeline)
        p.llm = SimpleNamespace(chat=AsyncMock())
        p.searcher = SimpleNamespace(chunks=[Chunk('S', '# S Title\n## First\nOriginal text'),
            Chunk('S', '# S Title\n## Second\n```\ncode()\n```'),
            Chunk('other', '# Other\nPrivate other section')],
            bm25=SimpleNamespace(scores=lambda query: [1, 3, 100]))
        s = State(messages=[{'role':'user','content':'private question'}], label=Label.NORMAL, section='S')
        await p._generate(s)
        self.assertLess(s.answer.index('## Second'), s.answer.index('## First'))
        self.assertIn('```\ncode()\n```', s.answer)
        self.assertIn('Original text', s.answer)
        self.assertNotIn('Private', s.answer)
        self.assertNotIn('private question', s.answer)
        self.assertEqual(s.citations, ['S'])
        p.llm.chat.assert_not_awaited()

    def test_all_bundled_sections_keep_content_and_fit_output_limit(self):
        p = Pipeline.__new__(Pipeline)
        p.searcher = Searcher(load_chunks(Path(__file__).resolve().parents[1] / 'kb'))
        for sid in p.searcher.section_ids():
            s = State(messages=[{'role':'user','content':'conditions and examples'}], section=sid)
            text = p._excerpt(s)
            self.assertEqual(Counter(text.split()), Counter(p.searcher.section_text(sid).split()))
            self.assertLessEqual(len('관련 교재 내용을 인용할게요.\n\n' + text), ANSWER_MAX_CHARS)

    def test_guard_and_policy_paths_keep_inherited_behavior(self):
        from pipelines import load
        self.assertIs(load('jh_r3_extract_focused'), Pipeline)
        for name in ('_generate', '_new_state', '_classify', '_search', '_check_output', 'guard'):
            self.assertIs(getattr(Pipeline, name), getattr(Section, name))

    def test_selector_compositions_keep_excerpt_hook_and_guard_identity(self):
        from pipelines import load
        for name, cls, parent in [('jh_r3_selected_focused', Selected, SelectedSection),
                                  ('jh_r3_grounded_focused', Grounded, GroundedSelected),
                                  ('jh_r3_quote_selected_focused', Quoted, QuoteSelected)]:
            self.assertIs(load(name), cls)
            self.assertIs(cls._excerpt, Pipeline._excerpt)
            self.assertIs(cls._generate, parent._generate)
            self.assertIs(cls._validate_source, parent._validate_source)
            self.assertEqual(cls.tutor_extract_scope, 'relevance_ordered_section')
            p = cls.__new__(cls)
            state = p._new_state({'messages': []}, True)
            self.assertTrue(state._extract_guard)
            self.assertTrue(state._section_gate_guard)
            for method in ('guard', '_check_output', '_classify'):
                self.assertIs(getattr(cls, method), getattr(parent, method))


if __name__ == '__main__':
    unittest.main()
