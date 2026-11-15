import asyncio
import time
import unittest

from core.schema import Label
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from pipelines.kimjaehak.voted import Pipeline
from pipelines.v2 import prompts
from tests.test_pipeline import FakeLLM, QUESTION
from tests.test_pipeline_v2 import SEARCHER, NONE_OTHER, FOUND


class VotedTests(unittest.TestCase):
    def test_failed_votes_do_not_outvote_one_valid_harmful_result(self):
        class PartialLLM(FakeLLM):
            failures = 0
            async def chat(self, system, user, **kwargs):
                if system == BLOCK_SYSTEM:
                    self.calls.append(system)
                    self.failures += 1
                    if self.failures < 3:
                        raise TimeoutError()
                    return '{"category":"harmful_request","confidence":90}'
                return await super().chat(system, user, **kwargs)
        llm = PartialLLM({prompts.OTHER_SYSTEM: NONE_OTHER, prompts.MATERIAL_SYSTEM: FOUND})
        state = asyncio.run(Pipeline(SEARCHER, llm).guard({"messages": QUESTION}))
        self.assertEqual(state.label, Label.HARMFUL_REQUEST)
        self.assertEqual(state.debug["valid_block_votes"], 1)
        self.assertEqual(llm.calls.count(prompts.MATERIAL_SYSTEM), 1)
        self.assertEqual(len(llm.calls), 7)

    def test_concurrency_queue_counts_toward_deadline_and_releases_slots(self):
        class Slow(Pipeline):
            request_slots = 1
            deadline = 0.025
            active = 0
            peak = 0
            async def _prepare_within(self, state, extra=""):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.1)
                finally:
                    self.active -= 1
        async def run():
            pipeline = Slow(SEARCHER, FakeLLM({}))
            start = time.monotonic()
            states = await asyncio.gather(*(pipeline.guard({"messages": QUESTION}) for _ in range(3)))
            self.assertLess(time.monotonic() - start, 0.1)
            self.assertEqual(pipeline.peak, 1)
            self.assertEqual(pipeline.active, 0)
            self.assertTrue(all("request:Deadline" in s.errors for s in states))
            self.assertTrue(all(s.to_guard_response()["input"]["decision"] == "allow" for s in states))
        asyncio.run(run())

    def test_pure_gibberish_does_not_use_votes(self):
        llm = FakeLLM({})
        state = asyncio.run(Pipeline(SEARCHER, llm).guard({"messages": [{"role":"user","content":"ㅁㄴㅇㄹ"}]}))
        self.assertEqual(state.label, Label.LOW_SIGNAL_INPUT)
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
