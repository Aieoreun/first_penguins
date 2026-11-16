import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import FALLBACK_ANSWER, Label
from core.state import State
from pipelines.kimjaehak.answer_choice import ContextFast, Pair, Pipeline, choose


class AnswerChoiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline = Pipeline.__new__(Pipeline)
        self.pipeline.llm = SimpleNamespace(chat=AsyncMock(return_value='{"choice":1}'))
        self.pipeline.searcher = SimpleNamespace(section_text=lambda _: 'material')
        self.state = State(messages=[{'role':'user','content':'question'}],label=Label.NORMAL,section='PY-01-01')
        self.state.answer_choice_deadline = time.monotonic() + 25
        self.counter = 0

    async def generate(self, pipeline, state):
        state.answer = f'draft{self.counter}'
        self.counter += 1
        state.citations = [state.section]

    async def run_choice(self):
        async def generate(pipeline, state):
            await self.generate(pipeline, state)
        with patch.object(ContextFast,'_generate',generate):
            await self.pipeline._generate(self.state)

    async def test_selects_existing_answer_preserving_input_and_source(self):
        await self.run_choice()
        self.assertEqual(self.state.answer,'draft1')
        self.assertEqual(self.state.citations,['PY-01-01'])
        self.assertEqual(self.state.label,Label.NORMAL)
        self.assertEqual(self.counter,3)

    async def test_invalid_choice_preserves_primary_and_records_failure(self):
        self.pipeline.llm.chat.return_value='{"choice":true}'
        await self.run_choice()
        self.assertEqual(self.state.answer,'draft0')
        self.assertIn('answer_choice:ValueError',self.state.errors)

    async def test_identical_drafts_skip_selector(self):
        async def identical(p,s):s.answer,s.citations='same',[s.section]
        with patch.object(ContextFast,'_generate',identical):
            await self.pipeline._generate(self.state)
        self.pipeline.llm.chat.assert_not_awaited()
        self.assertEqual(self.state.answer,'same')

    async def test_policy_or_short_budget_uses_only_original_generation(self):
        self.state.label=Label.ANSWER_SEEKING
        await self.run_choice()
        self.assertEqual(self.counter,1)
        self.state.label=Label.NORMAL
        self.state.answer_choice_deadline=time.monotonic()+1
        await self.run_choice()
        self.assertEqual(self.counter,2)
        self.pipeline.llm.chat.assert_not_awaited()

    async def test_failed_primary_can_use_successful_alternative(self):
        async def generate(p,s):
            s.answer,s.citations=('valid',[s.section]) if s is not self.state else (FALLBACK_ANSWER,[])
        with patch.object(ContextFast,'_generate',generate):
            await self.pipeline._generate(self.state)
        self.assertEqual(self.state.answer,'valid')
        self.pipeline.llm.chat.assert_not_awaited()

    async def test_parent_cancellation_cleans_up_alternatives(self):
        entered=asyncio.Event()
        active=0
        async def never(p,s):
            nonlocal active
            active+=1
            if active==3:entered.set()
            try:await asyncio.Future()
            finally:active-=1
        with patch.object(ContextFast,'_generate',never):
            task=asyncio.create_task(self.pipeline._generate(self.state))
            await asyncio.wait_for(entered.wait(),1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(active,0)

    async def test_selector_rejects_noninteger_and_out_of_range(self):
        for text in ('{}','{"choice":-1}','{"choice":2}','{"choice":"0"}','{"choice":false}'):
            self.pipeline.llm.chat.return_value=text
            with self.assertRaises(ValueError):await choose(self.pipeline.llm,'kb',[],['a','b'],2)

    def test_guard_path_and_classification_unchanged(self):
        from pipelines import load
        self.assertIs(load('jh_r3_answer_choice'),Pipeline)
        self.assertIs(load('jh_r3_answer_pair'),Pair)
        for name in ('guard','_classify','_check_output','_search'):
            self.assertIs(getattr(Pipeline,name),getattr(ContextFast,name))
