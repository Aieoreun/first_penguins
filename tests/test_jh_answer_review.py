import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.answer_review import ContextFast, Pipeline, reviewed_answer


class AnswerReviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline = Pipeline.__new__(Pipeline)
        self.pipeline.llm = SimpleNamespace(chat=AsyncMock(return_value='{"change":true,"answer":"수정 답변"}'))
        self.pipeline.searcher = SimpleNamespace(section_text=lambda _: "교재 원문")
        self.state = State(messages=[{"role":"user","content":"질문"}], label=Label.NORMAL,
                           section="PY-01-01", answer="기존 답변", citations=["PY-01-01"])
        self.state.answer_review_deadline = time.monotonic() + 20

    async def run_review(self):
        with patch.object(ContextFast, "_generate", new_callable=AsyncMock):
            await self.pipeline._generate(self.state)

    async def test_edit_preserves_source_and_label(self):
        await self.run_review()
        self.assertEqual(self.state.answer, "수정 답변 (PY-01-01)")
        self.assertEqual(self.state.citations, ["PY-01-01"])
        self.assertEqual(self.state.label, Label.NORMAL)
        self.assertEqual(self.pipeline.llm.chat.call_args.kwargs["timeout"], 7)

    async def test_unchanged_draft_is_preserved_exactly(self):
        self.pipeline.llm.chat.return_value = '{"change":false}'
        await self.run_review()
        self.assertEqual(self.state.answer, "기존 답변")

    async def test_invalid_json_and_timeout_preserve_draft(self):
        for response in ('{}', '{"change":"false"}', '{"change":true,"answer":""}'):
            self.pipeline.llm.chat.return_value = response
            await self.run_review()
            self.assertEqual(self.state.answer, "기존 답변")
        self.pipeline.llm.chat.side_effect = asyncio.TimeoutError
        await self.run_review()
        self.assertEqual(self.state.answer, "기존 답변")
        self.assertIn("answer_review:TimeoutError", self.state.errors)

    async def test_policy_paths_and_failed_drafts_are_not_reviewed(self):
        for label in (Label.ACADEMIC_DISHONESTY, Label.ANSWER_SEEKING, Label.HARMFUL_REQUEST, Label.OUT_OF_MATERIAL):
            self.state.label = label
            await self.run_review()
        self.state.label = Label.NORMAL
        self.state.citations = []
        await self.run_review()
        self.pipeline.llm.chat.assert_not_awaited()

    async def test_little_remaining_time_skips_review(self):
        self.state.answer_review_deadline = time.monotonic() + 1
        await self.run_review()
        self.pipeline.llm.chat.assert_not_awaited()

    async def test_cancellation_propagates(self):
        self.pipeline.llm.chat.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.run_review()

    async def test_rejects_overlong_or_non_string_answer(self):
        for answer in ([], "x" * 3901):
            import json
            llm = SimpleNamespace(chat=AsyncMock(return_value=json.dumps({"change":True,"answer":answer})))
            with self.assertRaises(ValueError):
                await reviewed_answer(llm,"material",[],"draft",3)

    def test_loader_and_guard_are_unchanged(self):
        from pipelines import load
        self.assertIs(load("jh_r2_answer_review"), Pipeline)
        for name in ("guard", "_classify", "_check_output"):
            self.assertIs(getattr(Pipeline,name),getattr(ContextFast,name))
