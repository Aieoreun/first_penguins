import asyncio
import unittest

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.reconsider import Pipeline, REVIEW_SYSTEM
from tests.test_pipeline import FakeLLM
from tests.test_pipeline_v2 import SEARCHER


class BlockReviewTests(unittest.TestCase):
    def test_failed_or_nonboolean_review_preserves_block(self):
        for response in (TimeoutError(), '{"benign":"true"}', '{"benign":false}'):
            state = State(messages=[], label=Label.JAILBREAK_INJECTION)
            asyncio.run(Pipeline(SEARCHER, FakeLLM({REVIEW_SYSTEM: response}))._review_block(state))
            self.assertEqual(state.label, Label.JAILBREAK_INJECTION)

    def test_benign_review_preserves_other_policy_flags(self):
        state = State(messages=[], label=Label.JAILBREAK_INJECTION,
                      debug={"other_votes":["emotional_distress"]*3, "self_info_votes":[False], "out_of_material":True})
        asyncio.run(Pipeline(SEARCHER, FakeLLM({REVIEW_SYSTEM:'{"benign":true}'}))._review_block(state))
        self.assertEqual(state.label, Label.EMOTIONAL_DISTRESS)
        self.assertEqual(state.errors, [])


if __name__ == "__main__":
    unittest.main()
