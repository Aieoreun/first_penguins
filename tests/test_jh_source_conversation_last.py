import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import Chunk
from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.source_conversation_last import SourceConversationLast


class SourceOrderTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self, cls=SourceConversationLast, guard=False):
        p=cls.__new__(cls)
        chunks=[Chunk(sid,f'# {sid} 제목\n## 문단\n{sid} 원문 “그대로”') for sid in ('A','B','C')]
        p.searcher=SimpleNamespace(chunks=chunks,bm25=SimpleNamespace(scores=lambda q:[3,2,1]),
            search=lambda *a,**kw:[chunks[1]],section_text=lambda sid:next(c.text for c in chunks if c.section_id==sid))
        p.llm=SimpleNamespace(chat=AsyncMock(return_value='{"section":"B"}'))
        s=p._new_state({'messages':[{'role':'user','content':'앞 질문'},
            {'role':'assistant','content':'앞 답변'}, {'role':'user','content':'현재 대상·조건 질문'}]},guard)
        s.label=Label.NORMAL;s.section='A';s.chunks=[chunks[0],chunks[2]]
        s._source_extract_deadline=108
        return p,s

    def test_baseline_payload_serialization_is_byte_identical(self):
        p,s=self.setup_case(SelectedFocused)
        ids=p._source_candidates(s)
        original={'conversation':s.messages,
                  'materials':[{'section':sid,'text':p.searcher.section_text(sid)} for sid in ids]}
        self.assertEqual(json.dumps(p._source_payload(s,ids),ensure_ascii=False).encode(),
                         json.dumps(original,ensure_ascii=False).encode())
        self.assertEqual(ids,['A','B','C'])

    def test_candidate_only_changes_top_level_key_order(self):
        p,s=self.setup_case()
        ids=p._source_candidates(s)
        baseline=SelectedFocused._source_payload(p,s,ids)
        candidate=p._source_payload(s,ids)
        self.assertEqual(list(baseline),['conversation','materials'])
        self.assertEqual(list(candidate),['materials','conversation'])
        self.assertEqual(candidate,baseline)
        self.assertIs(candidate['conversation'],s.messages)
        self.assertEqual([list(x) for x in candidate['materials']], [['section','text']]*3)
        self.assertEqual([x['section'] for x in candidate['materials']],ids)
        self.assertNotEqual(json.dumps(candidate,ensure_ascii=False),json.dumps(baseline,ensure_ascii=False))

    async def test_actual_chat_payload_system_and_budget(self):
        calls=[];answers=[]
        for cls in (SelectedFocused,SourceConversationLast):
            p,s=self.setup_case(cls)
            with patch('pipelines.kimjaehak.source_extract.time.monotonic',return_value=100):
                await p._generate(s)
            p.llm.chat.assert_awaited_once()
            calls.append(p.llm.chat.await_args)
            answers.append((s.answer,s.citations,s.section,s.errors))
        baseline,candidate=calls
        self.assertEqual(baseline.args[0],candidate.args[0])
        self.assertEqual(json.loads(baseline.args[1]),json.loads(candidate.args[1]))
        self.assertEqual(list(json.loads(candidate.args[1])),['materials','conversation'])
        self.assertEqual(baseline.kwargs,candidate.kwargs)
        self.assertEqual(candidate.kwargs,{'effort':'low','max_tokens':2000,'timeout':7.5,'json_mode':True})
        self.assertEqual(answers[0],answers[1])
        self.assertEqual(answers[1][1:],(['B'],'B',[]))

    async def test_guard_and_policy_bypass_payload_and_selector(self):
        for guard,label in ((True,Label.NORMAL),(False,Label.ANSWER_SEEKING)):
            p,s=self.setup_case(guard=guard);s.label=label;s.debug={}
            with (patch.object(p,'_source_payload',side_effect=AssertionError('must bypass')),
                  patch.object(ContextFast,'_generate',new_callable=AsyncMock) as original):
                await p._generate(s)
                original.assert_awaited_once_with(s)
            p.llm.chat.assert_not_awaited()

    async def test_high_gates_low_budget_and_single_source_keep_excerpt_fallback(self):
        for reason in ('learned_section','tutor_section_gate','budget','one_source'):
            p,s=self.setup_case()
            if reason in ('learned_section','tutor_section_gate'):s.debug[reason]=True
            if reason=='budget':s._source_extract_deadline=100.5
            if reason=='one_source':p._candidate_sections=AsyncMock(return_value=['A'])
            with patch('pipelines.kimjaehak.source_extract.time.monotonic',return_value=100),patch.object(p,'_source_payload',side_effect=AssertionError('must bypass')):
                await p._generate(s)
            p.llm.chat.assert_not_awaited()
            self.assertEqual(s.citations,['A'])

    async def test_invalid_source_and_cancellation_are_unchanged(self):
        p,s=self.setup_case();p.llm.chat.return_value='{"section":"unknown"}'
        with patch('pipelines.kimjaehak.source_extract.time.monotonic',return_value=100):await p._generate(s)
        self.assertEqual(s.citations,['A'])
        self.assertEqual(s.errors,['source_extract:ValueError'])
        self.assertEqual(p.llm.chat.await_count,1)
        p,s=self.setup_case();p.llm.chat.side_effect=asyncio.CancelledError()
        with patch('pipelines.kimjaehak.source_extract.time.monotonic',return_value=100):
            with self.assertRaises(asyncio.CancelledError):await p._generate(s)
        self.assertEqual(p.llm.chat.await_count,1)

    def test_guard_identity_and_all_other_components_are_inherited(self):
        for method in ('_new_state','guard','_classify','_generate','_check_output','_source_candidates',
                       '_candidate_sections','_validate_source','_excerpt'):
            self.assertIs(getattr(SourceConversationLast,method),getattr(SelectedFocused,method))
        for attr in ('source_extract_system','source_extract_sections','source_extract_timeout',
                     'deadline','prepare_limit','learned_gate_threshold','tutor_section_threshold'):
            self.assertEqual(getattr(SourceConversationLast,attr),getattr(SelectedFocused,attr))
        p,s=self.setup_case(guard=True)
        self.assertTrue(s._extract_guard)
        self.assertTrue(s._section_gate_guard)


if __name__=='__main__':
    unittest.main()
