import asyncio
import unittest

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.brief_material import MATERIAL_SYSTEM, Pipeline
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from pipelines.v2 import prompts
from tests.test_pipeline import FakeLLM, QUESTION
from tests.test_pipeline_v2 import SEARCHER, NONE_OTHER


class BriefMaterialTests(unittest.TestCase):
    def test_empty_quote_still_routes_out_of_material(self):
        llm = FakeLLM({BLOCK_SYSTEM:'{"category":"none"}', prompts.OTHER_SYSTEM:NONE_OTHER,
                       MATERIAL_SYSTEM:'{"quote":""}'})
        state = asyncio.run(Pipeline(SEARCHER, llm).guard({"messages":QUESTION}))
        self.assertEqual(state.label, Label.OUT_OF_MATERIAL)
        self.assertEqual(len(llm.calls), 5)
        self.assertEqual(state.errors, [])

    def test_invalid_quote_is_a_recorded_stage_failure(self):
        for reply in ('{"quote":null}', '{', TimeoutError()):
            state = State(messages=QUESTION)
            pipe = Pipeline(SEARCHER, FakeLLM({MATERIAL_SYSTEM:reply}))
            obj = asyncio.run(pipe._ask(state, "material", "unused", "material", "low", 10))
            self.assertEqual(obj, {})
            self.assertEqual(len(state.errors), 1)


if __name__ == "__main__":
    unittest.main()
