import asyncio
import json
import unittest

from core.schema import FALLBACK_ANSWER, Label
from core.state import Chunk, State
from pipelines.kimjaehak.multiground import Pipeline
from tests.test_pipeline_v2 import SEARCHER


class SourceChoiceTests(unittest.TestCase):
    def test_selected_source_controls_citation_and_invented_source_is_rejected(self):
        ids = SEARCHER.section_ids()[:3]
        chunks = [Chunk(sid, SEARCHER.section_text(sid)) for sid in ids]
        class LLM:
            chosen = ids[1]
            async def chat(self, system, user, **kwargs):
                payload = json.loads(user)
                self.supplied = [m["section_id"] for m in payload["materials"]]
                return json.dumps({"section": self.chosen, "answer": "자료의 조건을 비교해요."})
        llm = LLM()
        pipe = Pipeline(SEARCHER, llm)
        state = State(messages=[], label=Label.NORMAL, section=ids[0], chunks=chunks)
        asyncio.run(pipe._generate(state))
        self.assertEqual(llm.supplied, ids)
        self.assertEqual(state.section, ids[1])
        self.assertEqual(state.citations, [ids[1]])
        self.assertIn(ids[1], state.answer)
        self.assertEqual(state.errors, [])
        llm.chosen = "INVENTED-00"
        asyncio.run(pipe._generate(state))
        self.assertEqual(state.citations, [])
        self.assertEqual(state.answer, FALLBACK_ANSWER)
        self.assertEqual(state.errors, ["generate:ValueError"])


if __name__ == "__main__":
    unittest.main()
