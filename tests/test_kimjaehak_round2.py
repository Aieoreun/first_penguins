import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.search import Searcher
from core.state import Chunk, State
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.round2 import SectionFusion, Sources, TutorExamples
from pipelines.kimjaehak.fewshot import normalized
from tests.test_pipeline_v2 import SEARCHER


class Round2Tests(unittest.TestCase):
    def test_fusion_falls_back_exactly_when_embeddings_missing(self):
        for vector in (None, [1., 0.]):
            base = Searcher(SEARCHER.chunks)
            self.assertEqual(SectionFusion(base).search("파이썬", vector), base.search("파이썬"))

    def test_fusion_gives_sections_one_vote_each(self):
        chunks = [Chunk("A", "같은 토큰"), Chunk("A", "같은 토큰"), Chunk("B", "다른 내용")]
        base = Searcher(chunks, [[1., 0.], [1., 0.], [0., 1.]])
        found = SectionFusion(base).search("같은 토큰", [0., 1.])
        self.assertEqual({c.section_id for c in found}, {"A", "B"})
        self.assertEqual(len(found), 2)
        self.assertEqual(SectionFusion(base).search("같은 토큰", None), base.search("같은 토큰"))

    def test_examples_exclude_normalized_current_question_and_do_not_mutate(self):
        examples = TutorExamples()
        before = json.dumps(examples.items, sort_keys=True)
        query = examples.queries[0]
        messages = [{"role": "user", "content": query + "!!"}]
        selected = examples.select(messages, "answer")
        self.assertTrue(selected)
        self.assertTrue(all(normalized(row["question"]) != normalized(query) for row in selected))
        self.assertEqual(json.dumps(examples.items, sort_keys=True), before)

    def test_source_choice_uses_valid_citation_and_recovers_invalid_source(self):
        pipe = Sources.__new__(Sources)
        pipe.searcher, pipe.tutor_examples = SEARCHER, None
        ids = SEARCHER.section_ids()[:3]
        pipe.llm = type("LLM", (), {"chat": AsyncMock(return_value=json.dumps({"section":ids[1], "answer":"교재의 사실입니다."}))})()
        state = State(messages=[], label=Label.NORMAL, section=ids[0], chunks=[Chunk(i, "자료") for i in ids])
        asyncio.run(pipe._generate(state))
        self.assertEqual(state.citations, [ids[1]])
        pipe.llm.chat.return_value = '{"section":"invented","answer":"다른 답"}'
        with patch.object(ContextFast, "_generate", new_callable=AsyncMock) as fallback:
            asyncio.run(pipe._generate(state))
            fallback.assert_awaited_once_with(state)
        self.assertEqual(state.section, ids[1])
        self.assertEqual(state.errors, ["round2_generate:ValueError"])

    def test_high_confidence_and_policy_paths_preserve_generation(self):
        pipe = Sources.__new__(Sources)
        state = State(messages=[], label=Label.NORMAL, debug={"learned_section":True})
        with patch.object(ContextFast, "_generate", new_callable=AsyncMock) as fallback:
            asyncio.run(pipe._generate(state))
            state.debug.clear()
            state.label = Label.ACADEMIC_DISHONESTY
            asyncio.run(pipe._generate(state))
            self.assertEqual(fallback.await_count, 2)


if __name__ == "__main__":
    unittest.main()
