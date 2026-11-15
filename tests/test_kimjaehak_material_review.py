import asyncio
import json
import unittest

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.material_review import Pipeline, REVIEW_SYSTEM
from tests.test_pipeline import FakeLLM
from tests.test_pipeline_v2 import SEARCHER, TOP1


class MaterialReviewTests(unittest.TestCase):
    def state(self):
        return State(messages=[], label=Label.OUT_OF_MATERIAL, section=TOP1,
                     chunks=[Chunk(TOP1, SEARCHER.section_text(TOP1))], debug={"self_info_votes":[True,True,False]})

    def test_only_exact_evidence_can_restore_an_answerable_category(self):
        state = self.state()
        quote = SEARCHER.section_text(TOP1)[-40:]
        llm = FakeLLM({REVIEW_SYSTEM: json.dumps({"supported":True,"section":TOP1,"quote":quote})})
        asyncio.run(Pipeline(SEARCHER, llm)._review_material(state))
        self.assertEqual(state.label, Label.PII_DISCLOSURE)
        self.assertEqual(state.errors, [])

    def test_invented_evidence_and_timeouts_preserve_abstention(self):
        for response in (json.dumps({"supported":True,"section":TOP1,"quote":"교재에 존재하지 않는 근거 918273"}), TimeoutError()):
            state = self.state()
            llm = FakeLLM({REVIEW_SYSTEM: response})
            asyncio.run(Pipeline(SEARCHER, llm)._review_material(state))
            self.assertEqual(state.label, Label.OUT_OF_MATERIAL)
            self.assertEqual(len(state.errors), 1)


if __name__ == "__main__":
    unittest.main()
