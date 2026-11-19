import asyncio
import unittest

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.token_output import Pipeline, TOKEN_SYSTEM, exact_ranges
from tests.test_pipeline import FakeLLM
from tests.test_pipeline_v2 import SEARCHER, TOP1


class TokenOutputTests(unittest.TestCase):
    def test_ranges_preserve_exact_original_spacing_and_merge_overlap(self):
        spans, invalid = exact_ranges("첫째  둘째\n셋째 넷째", [[2,3],[3,4],[2,3]])
        self.assertEqual(spans, ["둘째\n셋째 넷째"])
        self.assertFalse(invalid)

    def test_invalid_ranges_are_not_allowed_to_create_substrings(self):
        spans, invalid = exact_ranges("하나 둘 셋", [[0,2],[1,9],[True,2],[2,1],[2,2]])
        self.assertEqual(spans, ["둘"])
        self.assertTrue(invalid)

    def test_output_uses_one_model_call_and_returns_word_range(self):
        llm = FakeLLM({TOKEN_SYSTEM:'{"unsupported_ranges":[[2,3]]}'})
        state = State(messages=[], label=Label.NORMAL, section=TOP1, candidate_answer="기간은 12일로 고정돼 있어요.")
        asyncio.run(Pipeline(SEARCHER, llm)._check_output(state))
        self.assertEqual(state.unsupported, ["12일로 고정돼"])
        self.assertFalse(state.grounded)
        self.assertEqual(state.errors, [])
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
