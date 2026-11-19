import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.answer_support import ContextFast, Pipeline
from pipelines.kimjaehak.plain_abstain import ABSTAIN


class SupportTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self, raw):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.searcher = type('KB', (), {'section_text': lambda self, sid: '한도는 10만 원이다.'})()
        pipeline.llm = type('LLM', (), {'chat': AsyncMock(return_value=raw)})()
        state = State(messages=[{'role': 'user', 'content': '한도는?'}], label=Label.NORMAL, section='FIN-01-01')
        state._support_deadline = time.monotonic() + 6
        return pipeline, state

    async def test_supported_keeps_answer_and_source(self):
        p, s = self.setup_case(json.dumps({'supported': True, 'answer': '10만 원이에요.'}))
        await p._generate(s)
        self.assertEqual(s.to_tutor_response(), {'action': 'answer', 'answer': '10만 원이에요. (FIN-01-01)', 'citations': ['FIN-01-01']})
        self.assertLessEqual(p.llm.chat.call_args.kwargs['timeout'], 5.5)

    async def test_unsupported_sets_abstention_without_invented_detail(self):
        p, s = self.setup_case(json.dumps({'supported': False, 'answer': '아마 15만 원일 수 있어요.'}))
        await p._generate(s)
        self.assertEqual(s.to_tutor_response(), {'action': 'abstain', 'answer': ABSTAIN, 'citations': []})

    async def test_invalid_decision_restores_original_prompt(self):
        p, s = self.setup_case('')
        p.llm.chat.side_effect = ['{"supported":"false","answer":"no"}', '10만 원이에요.']
        await p._generate(s)
        self.assertEqual(s.label, Label.NORMAL)
        self.assertEqual(s.errors, ['answer_support:ValueError'])
        self.assertEqual(p.llm.chat.call_args_list[1].args[0], ContextFast.answer_systems[Label.NORMAL])
        self.assertNotIn('json_mode', p.llm.chat.call_args_list[1].kwargs)
        self.assertTrue(s.answer.startswith('10만 원'))

    async def test_policy_and_guard_paths_bypass_support(self):
        for label, guard in [(Label.ANSWER_SEEKING, False), (Label.NORMAL, True)]:
            p, s = self.setup_case('{}')
            s.label, s._support_guard = label, guard
            with patch.object(ContextFast, '_generate', new_callable=AsyncMock) as generate:
                await p._generate(s)
                generate.assert_awaited_once_with(s)
            p.llm.chat.assert_not_awaited()

    async def test_cancellation_is_not_retried(self):
        p, s = self.setup_case('{}')
        p.llm.chat.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await p._generate(s)
        self.assertEqual(p.llm.chat.await_count, 1)
        self.assertFalse(s.errors)

    def test_guard_helpers_and_registry(self):
        from pipelines import load
        self.assertIs(load('jh_r3_answer_support'), Pipeline)
        for method in ('guard', '_classify', '_search', '_check_output'):
            self.assertIs(getattr(Pipeline, method), getattr(ContextFast, method))
