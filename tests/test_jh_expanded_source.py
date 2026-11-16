import asyncio
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.extractive_answer import Section
from pipelines.kimjaehak.focused_excerpt import Expanded, Selected
from pipelines.kimjaehak.source_extract import ExpandedSelected, SelectedSection


class ExpandedSourceTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self, chunks=('old', 'A', 'B', 'C', 'D', 'E'),
                   last=('last', 'old', 'X', 'X', 'Y', 'Z', 'Q', 'R', 'S', 'T')):
        p = ExpandedSelected.__new__(ExpandedSelected)
        hits = [Chunk(sid, sid + ' material') for sid in last]
        p.searcher = SimpleNamespace(
            search=Mock(side_effect=lambda query, bm25_top, embedding_top: hits[:bm25_top]),
            section_text=lambda sid: sid + ' source material')
        p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"section":"Y"}'))
        s = State(messages=[{'role': 'user', 'content': 'earlier question'},
                            {'role': 'assistant', 'content': 'earlier response'},
                            {'role': 'user', 'content': 'last question'}],
                  label=Label.NORMAL, section='old',
                  chunks=[Chunk(sid, sid + ' material') for sid in chunks])
        s._source_extract_deadline = time.monotonic() + 100
        return p, s

    def test_base_helper_preserves_original_candidates_and_search(self):
        p, s = self.setup_case(chunks=('old', 'last', 'A', 'A', 'B', 'C', 'D', 'E'))
        self.assertEqual(SelectedSection._source_candidates(p, s),
                         ['old', 'last', 'A', 'B', 'C', 'D'])
        p.searcher.search.assert_called_once_with('last question', bm25_top=1, embedding_top=0)

    def test_expansion_preserves_base_prefix_deduplicates_and_caps_eight(self):
        p, s = self.setup_case()
        original_chunks = list(s.chunks)
        self.assertEqual(p._source_candidates(s), ['old', 'last', 'A', 'B', 'C', 'D', 'X', 'Y'])
        self.assertEqual(s.chunks, original_chunks)
        self.assertEqual(s.section, 'old')
        self.assertEqual(p.searcher.search.call_args_list, [
            call('last question', bm25_top=1, embedding_top=0),
            call('last question', bm25_top=10, embedding_top=0)])

    def test_short_base_can_append_multiple_sections_and_duplicates_do_not_fill_slots(self):
        p, s = self.setup_case(chunks=('old',), last=('last', 'last', 'A', 'B', 'C', 'D', 'E', 'F', 'G'))
        self.assertEqual(p._source_candidates(s), ['old', 'last', 'A', 'B', 'C', 'D', 'E', 'F'])
        p, s = self.setup_case(chunks=('old', 'A'), last=('old', 'old', 'A', 'A'))
        self.assertEqual(p._source_candidates(s), ['old', 'A'])

    async def test_appended_section_is_selectable_with_one_original_budgeted_call(self):
        for label in (Label.NORMAL, Label.PII_DISCLOSURE):
            p, s = self.setup_case()
            s.label = label
            await p._generate(s)
            p.llm.chat.assert_awaited_once()
            args, kwargs = p.llm.chat.call_args
            self.assertEqual(args[0], SelectedSection.source_extract_system)
            payload = json.loads(args[1])
            self.assertEqual(payload['conversation'], s.messages)
            self.assertEqual([x['section'] for x in payload['materials']],
                             ['old', 'last', 'A', 'B', 'C', 'D', 'X', 'Y'])
            self.assertEqual(kwargs, dict(effort='low', max_tokens=2000, timeout=8, json_mode=True))
            self.assertEqual(s.section, 'Y')
            self.assertEqual(s.citations, ['Y'])
            self.assertEqual(s.answer, '관련 교재 내용을 인용할게요.\n\nY source material')

    async def test_guard_policy_and_section_gates_bypass_expansion_and_selector(self):
        cases = [(True, Label.NORMAL, {}, 'old'),
                 (False, Label.NORMAL, {}, None),
                 (False, Label.NORMAL, {'learned_section': True}, 'old'),
                 (False, Label.NORMAL, {'tutor_section_gate': .90}, 'old')]
        cases += [(False, label, {}, 'old') for label in Label
                  if label not in {Label.NORMAL, Label.PII_DISCLOSURE}]
        for guard, label, debug, section in cases:
            with self.subTest(guard=guard, label=label, debug=debug, section=section):
                p, s = self.setup_case()
                s._extract_guard, s.label, s.debug, s.section = guard, label, debug, section
                with patch.object(Section, '_generate', new_callable=AsyncMock) as inherited:
                    await p._generate(s)
                    inherited.assert_awaited_once_with(s)
                p.searcher.search.assert_not_called()
                p.llm.chat.assert_not_awaited()

    async def test_invalid_and_failed_selection_keep_original_fallback_without_retry(self):
        for raw, error in [('{"section":"E"}', None), (None, TimeoutError())]:
            p, s = self.setup_case()
            p.llm.chat.return_value, p.llm.chat.side_effect = raw, error
            await p._generate(s)
            self.assertEqual(s.citations, ['old'])
            self.assertEqual(s.errors, ['source_extract:' + ('TimeoutError' if error else 'ValueError')])
            p.llm.chat.assert_awaited_once()

    async def test_cancellation_propagates_and_expired_deadline_makes_no_llm_call(self):
        p, s = self.setup_case()
        p.llm.chat.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await p._generate(s)
        p.llm.chat.assert_awaited_once()
        self.assertEqual(s.errors, [])
        p, s = self.setup_case()
        s._source_extract_deadline = time.monotonic()
        await p._generate(s)
        p.llm.chat.assert_not_awaited()
        self.assertEqual(s.citations, ['old'])

    def test_expanded_composition_keeps_guard_prepare_gate_and_budget_settings(self):
        for name in ('guard', '_classify', '_check_output', '_prepare_within', '_new_state', '_generate'):
            self.assertIs(getattr(Expanded, name), getattr(Selected, name))
        for name in ('deadline', 'prepare_limit', 'source_extract_timeout',
                     'source_extract_sections', 'tutor_section_threshold', 'learned_section_threshold'):
            self.assertEqual(getattr(Expanded, name), getattr(Selected, name))
        self.assertEqual(Expanded.source_extract_sections, 6)
        self.assertEqual(Expanded.source_extract_expanded_limit, 8)

    def test_server_records_distinct_original_and_expanded_candidate_limits(self):
        from experiments import jh_server
        run = Mock()
        with patch.dict(os.environ, {key: 'test-only' for key in
                        ('TEAM_TOKEN', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL')}), \
                patch.dict('sys.modules', {'uvicorn': SimpleNamespace(run=run)}), \
                patch('dotenv.load_dotenv'), \
                patch('sys.argv', ['jh_server.py', '--variant', 'jh_r3_expanded_focused']), \
                patch('builtins.print') as output:
            jh_server.main()
        metadata = json.loads(output.call_args.args[0].removeprefix('EXPERIMENT_CONFIG '))
        self.assertEqual(metadata['class'], 'pipelines.kimjaehak.focused_excerpt.Expanded')
        self.assertEqual(metadata['source_extract_sections'], 6)
        self.assertEqual(metadata['source_extract_expanded_limit'], 8)
        self.assertEqual(metadata['source_extract_timeout'], 8)
        run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
