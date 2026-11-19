import asyncio
import itertools
import unittest

from core.schema import Label
from pipelines.kimjaehak.adaptive import Pipeline, Unanimous, needs_third
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from pipelines.v2 import prompts
from pipelines.v2.pipeline import _majority
from tests.test_pipeline import FakeLLM, QUESTION
from tests.test_pipeline_v2 import SEARCHER, NONE_OTHER, FOUND


class AdaptiveTests(unittest.TestCase):
    def test_skipping_a_vote_cannot_change_plurality_for_any_third_value(self):
        values = (None, "none", "harmful_request", "jailbreak_injection")
        for pair in itertools.product(values, repeat=2):
            answers = [{"category": value} for value in pair]
            if not needs_third(answers):
                valid = [value for value in pair if value is not None]
                for third in values:
                    after = valid + ([third] if third is not None else [])
                    self.assertEqual(_majority(valid), _majority(after))

    def test_personal_disclosure_disagreement_requires_third_vote(self):
        self.assertTrue(needs_third([{"category":"none","self_info":True},
                                    {"category":"none","self_info":False}], other=True))

    def test_agreeing_majorities_use_five_calls(self):
        llm = FakeLLM({BLOCK_SYSTEM:'{"category":"none"}', prompts.OTHER_SYSTEM:NONE_OTHER,
                       prompts.MATERIAL_SYSTEM:FOUND})
        state = asyncio.run(Pipeline(SEARCHER, llm).guard({"messages":QUESTION}))
        self.assertEqual(state.label, Label.NORMAL)
        self.assertEqual(state.errors, [])
        self.assertEqual(len(llm.calls), 5)
        self.assertEqual(state.debug["block_calls"], 2)

    def test_unanimous_block_still_requires_the_third_matching_vote(self):
        llm = FakeLLM({BLOCK_SYSTEM:'{"category":"harmful_request"}', prompts.OTHER_SYSTEM:NONE_OTHER,
                       prompts.MATERIAL_SYSTEM:FOUND})
        state = asyncio.run(Unanimous(SEARCHER, llm).guard({"messages":QUESTION}))
        self.assertEqual(state.label, Label.HARMFUL_REQUEST)
        self.assertEqual(state.errors, [])
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(state.debug["block_calls"], 3)


if __name__ == "__main__":
    unittest.main()
