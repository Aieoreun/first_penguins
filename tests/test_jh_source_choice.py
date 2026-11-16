import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.source_choice import ContextFast, Pipeline


class SourceChoiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline = Pipeline.__new__(Pipeline)
        self.pipeline.searcher = object()
        self.pipeline.llm = SimpleNamespace(chat=AsyncMock(return_value='{"section":"b"}'))
        self.state = State(messages=[{"role":"user","content":"설명해주세요"}], section="a", label=Label.NORMAL)
        self.materials = patch("pipelines.kimjaehak.source_choice.materials",
                               return_value=[{"section":"a","text":"A"},{"section":"b","text":"B"}])
        self.materials.start()
        self.addCleanup(self.materials.stop)

    async def test_changed_source_preserves_input_and_uses_selected_answer(self):
        async def generate(state):
            state.answer, state.citations = "answer " + state.section, [state.section]
        with patch.object(ContextFast,"_generate",side_effect=generate):
            await self.pipeline._generate(self.state)
        self.assertEqual((self.state.section,self.state.answer,self.state.citations), ("b","answer b",["b"]))
        self.assertEqual(self.state.label,Label.NORMAL)

    async def test_invalid_choice_returns_original_answer(self):
        self.pipeline.llm.chat.return_value = '{"section":"not provided"}'
        async def generate(state):
            state.answer, state.citations = "original", [state.section]
        with patch.object(ContextFast,"_generate",side_effect=generate):
            await self.pipeline._generate(self.state)
        self.assertEqual((self.state.section,self.state.answer), ("a","original"))
        self.assertEqual(self.state.errors,["source_choice:ValueError"])

    async def test_confident_section_skips_rerank(self):
        self.state.debug["learned_section"] = True
        with patch.object(ContextFast,"_generate",new_callable=AsyncMock) as generate:
            await self.pipeline._generate(self.state)
            generate.assert_awaited_once_with(self.state)
        self.pipeline.llm.chat.assert_not_awaited()

    async def test_cancelled_request_awaits_backup_cleanup(self):
        started, closed = asyncio.Event(), asyncio.Event()
        async def generate(state):
            started.set()
            try:
                await asyncio.Future()
            finally:
                closed.set()
        async def choose(*args,**kwargs):
            await asyncio.Future()
        self.pipeline.llm.chat.side_effect = choose
        with patch.object(ContextFast,"_generate",side_effect=generate):
            task = asyncio.create_task(self.pipeline._generate(self.state))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(closed.is_set())

    async def test_failed_selected_answer_recovers_original(self):
        async def generate(state):
            if state.section == "b":
                state.errors.append("generate:TimeoutError")
            else:
                state.answer, state.citations = "original", [state.section]
        with patch.object(ContextFast,"_generate",side_effect=generate):
            await self.pipeline._generate(self.state)
        self.assertEqual((self.state.section,self.state.answer),("a","original"))
        self.assertEqual(self.state.errors,["generate:TimeoutError"])
