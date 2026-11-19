import asyncio
import json
import time
import unittest
from unittest.mock import patch

from core.schema import Label
from pipelines import load
load('jh_cm_char_mlp50')
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_answer import NumericApply, AllApply
from tests import test_jh_reasoning_ablation as reasoning


class AnswerTests(unittest.IsolatedAsyncioTestCase):
    pipeline = reasoning.ReasoningTests.pipeline
    state = reasoning.ReasoningTests.state

    def ready(self, cls=NumericApply):
        p = self.pipeline(cls)
        s = self.state(p, False)
        s.label = Label.NORMAL
        s.messages = [{'role': 'user', 'content': '이자는 얼마인가요?'}]
        s.answer = '관련 교재 내용을 인용할게요.\n\n원래 교재 전체'
        s.citations = ['A']
        s._source_extract_deadline = time.monotonic() + 20
        p.llm.chat.return_value = json.dumps({'supported': True, 'answer': '계산한 설명'})
        return p, s

    async def test_preserves_complete_excerpt_citations_and_bounds_call(self):
        p, s = self.ready()
        old = s.answer
        with patch.object(CharMLP50, '_generate', return_value=None) as base:
            await p._generate(s)
        base.assert_awaited_once_with(s)
        self.assertEqual(s.answer, '계산한 설명\n\n' + old)
        self.assertEqual(s.citations, ['A'])
        self.assertTrue(s.debug['r7_apply_completed'])
        call = p.llm.chat.await_args
        self.assertEqual(json.loads(call.args[1])['conversation'], s.messages)
        self.assertEqual(json.loads(call.args[1])['material'], p.searcher.section_text('A'))
        self.assertEqual(call.kwargs, dict(effort='low', max_tokens=2000, timeout=8, json_mode=True))

    async def test_skips_guard_nonanswers_nonnumeric_missing_section_and_long_excerpt(self):
        for mode in ('guard', 'seeking', 'nonnumeric', 'no_section', 'generated', 'long', 'budget'):
            p, s = self.ready()
            if mode == 'guard': s._material_budget_guard = True
            if mode == 'seeking': s.label = Label.ANSWER_SEEKING
            if mode == 'nonnumeric': s.messages[-1]['content'] = '리스트의 뜻을 설명해요'
            if mode == 'no_section': s.section = None
            if mode == 'generated': s.answer = '이미 작성한 일반 답변'
            if mode == 'long': s.answer += 'x' * 3200
            if mode == 'budget': s._source_extract_deadline = time.monotonic() + 3
            old = s.answer
            with patch.object(CharMLP50, '_generate', return_value=None): await p._generate(s)
            p.llm.chat.assert_not_awaited()
            self.assertEqual(s.answer, old)
        p, s = self.ready(AllApply)
        s.messages[-1]['content'] = '리스트의 뜻을 설명해요'
        with patch.object(CharMLP50, '_generate', return_value=None): await p._generate(s)
        p.llm.chat.assert_awaited_once()

    async def test_failure_unknown_or_invalid_response_keeps_original(self):
        for raw in ('{}', '{"supported":false,"answer":""}', '{"supported":"yes","answer":"x"}',
                    json.dumps({'supported':True, 'answer':'x'*501}), 'bad json', TimeoutError()):
            p, s = self.ready()
            old = s.answer
            if isinstance(raw, Exception): p.llm.chat.side_effect = raw
            else: p.llm.chat.return_value = raw
            with patch.object(CharMLP50, '_generate', return_value=None): await p._generate(s)
            self.assertEqual(s.answer, old)
            self.assertEqual(s.citations, ['A'])
            self.assertNotIn('r7_apply_completed', s.debug)

    async def test_deadline_clamps_and_cancellation_propagates(self):
        p, s = self.ready()
        s._source_extract_deadline = time.monotonic() + 5
        with patch.object(CharMLP50, '_generate', return_value=None): await p._generate(s)
        self.assertLessEqual(p.llm.chat.await_args.kwargs['timeout'], 4.5)
        p, s = self.ready()
        p.llm.chat.side_effect = asyncio.CancelledError()
        with patch.object(CharMLP50, '_generate', return_value=None):
            with self.assertRaises(asyncio.CancelledError): await p._generate(s)

    def test_a_and_guard_methods_preserved(self):
        self.assertIs(load('jh_cm_char_mlp50'), CharMLP50)
        for name, cls in (('jh_r7_apply_numeric', NumericApply), ('jh_r7_apply_all', AllApply)):
            self.assertIs(load(name), cls)
            for method in ('guard', '_classify', '_search', '_fusion_ground', '_fusion_safety'):
                self.assertIs(getattr(cls, method), getattr(CharMLP50, method))


if __name__ == '__main__': unittest.main()
