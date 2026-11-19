import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from core.kb import load_chunks
from core.search import Searcher
from core.schema import Label
from core.state import Chunk
from pipelines.kimjaehak.extractive_answer import Section
from pipelines.kimjaehak.last_dense_source import Pipeline as LastDense
from pipelines.kimjaehak.conditional_dense_source import Pipeline, build_anchors, _matcher, KB_DIR


class AnchorTests(unittest.TestCase):
    def test_heading_inline_only_length_and_full_section_rarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); subject = base / 'sample'; subject.mkdir()
            texts = {
                'A': '# A 다이아몬드 abc ab 사과 두글자\n## rare_key common_key\n`inline_key` `two_key`\nbody_only 본문전용표현\n```python\n## fenced_heading\nfenced_code\n```',
                'B': '# B 제목\ncommon_key two_key',
                'C': '# C 제목\ncommon_key'}
            for sid, text in texts.items(): (subject / (sid + '.md')).write_text(text)
            searcher = SimpleNamespace(section_ids=lambda: list(texts), section_text=texts.get)
            anchors = build_anchors(base, searcher)
            self.assertIn('다이아몬드', anchors)
            self.assertIn('abc', anchors)
            self.assertIn('inline_key', anchors)
            self.assertEqual(anchors['two_key'][1], frozenset({'A', 'B'}))
            for absent in ('ab', '사과', '두글자', 'common_key', 'body_only', '본문전용표현', 'fenced_heading', 'fenced_code'):
                self.assertNotIn(absent, anchors)

    def test_matching_keeps_english_identifier_boundaries_and_korean_suffix(self):
        english = _matcher('fruit_basket')
        self.assertIsNotNone(english.search('FRUIT_BASKET의 값'))
        for text in ('xfruit_basket', 'fruit_basket2', 'fruit_basket_extra'):
            self.assertIsNone(english.search(text))
        self.assertIsNotNone(_matcher('다이아몬드').search('다이아몬드의 모양'))



class ConditionalTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self, scores=(15, 14, 9)):
        p = Pipeline.__new__(Pipeline)
        p._lexical_anchors = {'anchor_name': (_matcher('anchor_name'), frozenset({'old'}))}
        p.searcher = SimpleNamespace(chunks=[Chunk('old', 'old'), Chunk('old', 'old second'), Chunk('last', 'last')],
            bm25=SimpleNamespace(scores=Mock(return_value=list(scores))), has_embeddings=True,
            search=Mock(side_effect=lambda text, vector=None, **kw: [Chunk(s, s) for s in (('D1', 'D2') if vector is not None else ('last',))]),
            section_text=lambda sid: sid + ' material')
        p.llm = SimpleNamespace(can_embed=True, embed=AsyncMock(return_value=[[.1, .9]]),
                                chat=AsyncMock(return_value='{"section":"old"}'))
        s = p._new_state({'messages': [{'role': 'user', 'content': 'anchor_name question'}]}, False)
        s.label, s.section = Label.NORMAL, 'old'
        s.chunks = [Chunk(sid, sid) for sid in ('old', 'A', 'B', 'C', 'E')]
        p._excerpt = lambda state: p.searcher.section_text(state.section)
        return p, s

    def test_distinct_section_margin_not_second_chunk_and_exact_boundary(self):
        p, s = self.make_case((15, 14, 10))
        self.assertTrue(p._lexical_strong(s.last_user))
        p.searcher.bm25.scores.return_value = [14.99, 14, 10]
        self.assertFalse(p._lexical_strong(s.last_user))
        p.searcher.bm25.scores.return_value = [15, 14, 15]
        self.assertFalse(p._lexical_strong(s.last_user))

    def test_zero_empty_unmatched_and_anchor_in_other_section_are_not_strong(self):
        for case in ('zero', 'empty', 'unmatched', 'other_section'):
            p, s = self.make_case()
            if case == 'zero': p.searcher.bm25.scores.return_value = [0, 0, 0]
            if case == 'empty': p.searcher.chunks = []
            if case == 'unmatched': s.messages[-1]['content'] = 'other query'
            if case == 'other_section': p._lexical_anchors['anchor_name'] = (_matcher('anchor_name'), frozenset({'last'}))
            self.assertFalse(p._lexical_strong(s.last_user))

    async def test_strong_preserves_exact_base_and_skips_embed_but_keeps_selector(self):
        p, s = self.make_case()
        expected = p._source_candidates(s)
        await p._generate(s)
        p.llm.embed.assert_not_awaited()
        p.llm.chat.assert_awaited_once()
        payload = json.loads(p.llm.chat.await_args.args[1])
        self.assertEqual([row['section'] for row in payload['materials']], expected)
        self.assertEqual(payload['conversation'], s.messages)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertEqual(p.llm.chat.await_args.kwargs['timeout'], 8)

    async def test_weak_calls_real_parent_embed_and_keeps_six_candidate_layout(self):
        p, s = self.make_case((10, 9, 8))
        self.assertEqual(await p._candidate_sections(s), ['old', 'last', 'D1', 'D2', 'A', 'B'])
        p.llm.embed.assert_awaited_once_with([s.last_user], 3.0)

    async def test_weak_embedding_failure_uses_parent_fallback_without_retry(self):
        p, s = self.make_case((10, 9, 8))
        p.llm.embed.side_effect = TimeoutError()
        expected = p._source_candidates(s)
        self.assertEqual(await p._candidate_sections(s), expected)
        p.llm.embed.assert_awaited_once()
        self.assertEqual(s.errors, ['source_last_dense:TimeoutError'])

    async def test_guard_policy_and_gates_bypass_inherited_hook(self):
        cases = [(True, Label.NORMAL, {}), (False, Label.ANSWER_SEEKING, {}),
                 (False, Label.NORMAL, {'learned_section': True}),
                 (False, Label.NORMAL, {'tutor_section_gate': .91})]
        for guard, label, debug in cases:
            p, s = self.make_case()
            s._extract_guard, s.label, s.debug = guard, label, debug
            with patch.object(Section, '_generate', new_callable=AsyncMock): await p._generate(s)
            p.searcher.bm25.scores.assert_not_called()
            p.searcher.search.assert_not_called()
            p.llm.embed.assert_not_awaited()
            p.llm.chat.assert_not_awaited()

    def test_constructor_builds_kb_anchors_once_and_inherits_policy_and_budgets(self):
        searcher, llm = object(), object()
        with patch.object(LastDense, '__init__', return_value=None) as parent, \
                patch('pipelines.kimjaehak.conditional_dense_source.build_anchors', return_value={'kb_only': 'value'}) as build:
            p = Pipeline(searcher, llm)
        parent.assert_called_once_with(searcher, llm)
        build.assert_called_once_with(KB_DIR, searcher)
        self.assertEqual(p._lexical_anchors, {'kb_only': 'value'})
        for method in ('_classify', 'guard', '_generate', '_search', '_check_output', '_run', '_new_state'):
            self.assertIs(getattr(Pipeline, method), getattr(LastDense, method))
        for attribute in ('deadline', 'prepare_limit', 'source_extract_sections', 'source_extract_timeout'):
            self.assertEqual(getattr(Pipeline, attribute), getattr(LastDense, attribute))


if __name__ == '__main__': unittest.main()
