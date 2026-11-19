import asyncio
import unittest

from core.state import Chunk, State
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from pipelines.kimjaehak.thin_material import Pipeline, AdaptivePipeline
from pipelines.v2 import prompts
from tests.test_pipeline import FakeLLM, QUESTION
from tests.test_pipeline_v2 import SEARCHER, NONE_OTHER, FOUND


class ThinMaterialTests(unittest.TestCase):
    def test_only_material_prompt_is_shortened(self):
        class CaptureLLM(FakeLLM):
            async def chat(self, system, user, **kwargs):
                if system == prompts.MATERIAL_SYSTEM:
                    self.material = user
                return await super().chat(system, user, **kwargs)
        llm = CaptureLLM({BLOCK_SYSTEM:'{"category":"none"}', prompts.OTHER_SYSTEM:NONE_OTHER,
                          prompts.MATERIAL_SYSTEM:FOUND})
        chunks = [Chunk(f"SECTION-{i}", f"UNIQUE_MATERIAL_{i}") for i in range(8)]
        state = State(messages=QUESTION, chunks=chunks)
        asyncio.run(Pipeline(SEARCHER, llm)._classify(state))
        self.assertIn("UNIQUE_MATERIAL_4", llm.material)
        self.assertNotIn("UNIQUE_MATERIAL_5", llm.material)
        self.assertEqual(state.chunks, chunks)
        self.assertEqual(len(state.chunks), 8)
        self.assertEqual(state.errors, [])
        self.assertTrue(AdaptivePipeline.adaptive_votes)
        self.assertEqual(AdaptivePipeline.material_max_chunks, 5)


if __name__ == "__main__":
    unittest.main()
