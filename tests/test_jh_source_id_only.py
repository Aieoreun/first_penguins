import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipelines.kimjaehak.source_id_only import Pipeline
from core.schema import Label
from core.search import Searcher
from core.state import Chunk
from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused
from pipelines.kimjaehak.guard_fitted import ContextFast


class SourceIDOnlyTests(unittest.IsolatedAsyncioTestCase):
    def make_case(self, cls=Pipeline, response='{"section":"S1"}', guard=False):
        p = cls.__new__(cls)
        p.llm = SimpleNamespace(chat=AsyncMock(return_value=response),
            _chat=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=AsyncMock(side_effect=AssertionError('Streaming/direct client forbidden'))))))
        p.searcher = Searcher([Chunk('S'+str(i), 'Title '+str(i)+'\nmaterial '+str(i)) for i in range(8)])
        s = p._new_state({'messages':[{'role':'user','content':'material 1'}]}, guard)
        s.label, s.section, s.chunks = Label.NORMAL, 'S0', p.searcher.chunks
        return p,s

    def test_only_final_schema_sentence_changes(self):
        before, after = SelectedFocused.source_extract_system, Pipeline.source_extract_system
        self.assertEqual(before.splitlines(keepends=True)[:-1], after.splitlines(keepends=True)[:-1])
        self.assertEqual(after.splitlines()[-1], 'JSON 하나: {"section":"제공된 섹션 ID 하나"}.')
        self.assertEqual(before.splitlines()[-1], 'JSON 하나: {"section":"제공된 섹션 ID 하나", "reason":"선택 이유를 짧은 한 문장으로"}.')
        self.assertNotIn('reason', after)

    async def test_nonstream_call_matches_baseline_except_system(self):
        p,s = self.make_case()
        base,bs = self.make_case(SelectedFocused)
        await p._generate(s)
        await base._generate(bs)
        p.llm.chat.assert_awaited_once()
        base.llm.chat.assert_awaited_once()
        actual, original = p.llm.chat.await_args, base.llm.chat.await_args
        self.assertEqual(actual.args[0], Pipeline.source_extract_system)
        self.assertEqual(original.args[0], SelectedFocused.source_extract_system)
        self.assertEqual(actual.args[1:], original.args[1:])
        self.assertEqual(actual.kwargs, original.kwargs)
        self.assertEqual(actual.kwargs, {'effort':'low','max_tokens':2000,'timeout':8,'json_mode':True})
        ids = [m['section'] for m in json.loads(actual.args[1])['materials']]
        self.assertEqual(ids, ['S0','S1','S2','S3','S4','S5'])
        self.assertEqual(s.to_tutor_response(), bs.to_tutor_response())
        self.assertEqual(s.section, 'S1')
        p.llm._chat.chat.completions.create.assert_not_awaited()

    async def test_full_parser_rejects_truncated_prefix_without_retry(self):
        p,s = self.make_case(response='{"section":"S1",')
        await p._generate(s)
        self.assertEqual(s.section, 'S0')
        self.assertEqual(s.errors, ['source_extract:ValueError'])
        self.assertEqual(s.citations, ['S0'])
        self.assertEqual(p.llm.chat.await_count, 1)
        p.llm._chat.chat.completions.create.assert_not_awaited()

    async def test_complete_json_uses_last_duplicate_and_validates_it(self):
        for response, selected, errors in [
            ('{"section":"S1","section":"S2"}', 'S2', []),
            ('{"section":"S1","section":"outside"}', 'S0', ['source_extract:ValueError']),
            ('{"section":"S1","reason":"extra is still accepted"}', 'S1', [])]:
            with self.subTest(response=response):
                p,s = self.make_case(response=response)
                await p._generate(s)
                self.assertEqual(s.section, selected)
                self.assertEqual(s.errors, errors)
                self.assertEqual(p.llm.chat.await_count, 1)

    async def test_guard_policy_and_section_gates_do_not_call_source(self):
        for mode in ('guard', 'policy', 'learned_section', 'tutor_section_gate', 'expired'):
            p,s = self.make_case(guard=mode=='guard')
            if mode=='policy': s.label=Label.ANSWER_SEEKING
            elif mode in ('learned_section','tutor_section_gate'): s.debug[mode]=True
            elif mode=='expired': s._source_extract_deadline=time.monotonic()
            with patch.object(ContextFast, '_generate', new_callable=AsyncMock):
                await p._generate(s)
            p.llm.chat.assert_not_awaited()
            p.llm._chat.chat.completions.create.assert_not_awaited()

    async def test_remaining_budget_and_cancel_stay_inherited(self):
        p,s = self.make_case()
        s._source_extract_deadline=time.monotonic()+2
        p.llm.chat.side_effect=asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await p._generate(s)
        self.assertGreater(p.llm.chat.await_args.kwargs['timeout'],1)
        self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'],1.5)
        self.assertEqual(s.errors,[])
        self.assertEqual(p.llm.chat.await_count,1)

    async def test_provider_timeout_preserves_original_excerpt(self):
        p,s=self.make_case()
        p.llm.chat.side_effect=TimeoutError()
        await p._generate(s)
        self.assertEqual(s.section,'S0')
        self.assertEqual(s.errors,['source_extract:TimeoutError'])
        self.assertEqual(p.llm.chat.await_count,1)

    def test_methods_and_budgets_are_baseline_unchanged(self):
        for method in ('guard','tutor','_new_state','_prepare_within','_ask','_source_response',
                       '_generate','_validate_source','_source_payload','_source_candidates','_candidate_sections'):
            self.assertIs(getattr(Pipeline,method),getattr(SelectedFocused,method))
        self.assertEqual((Pipeline.prepare_limit,Pipeline.source_extract_timeout,Pipeline.deadline,
                          Pipeline.source_extract_sections),(14,8,26,6))
        self.assertFalse(getattr(Pipeline,'source_stream_early_finish',False))
        for name in ('learned_gate_threshold','learned_section_threshold','tutor_section_threshold'):
            self.assertEqual(getattr(Pipeline,name),getattr(SelectedFocused,name))


if __name__=='__main__':
    unittest.main(verbosity=2)
