import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.medium_output import Pipeline
from tests.test_pipeline import FakeLLM
from tests.test_pipeline_v2 import SEARCHER, TOP1


class MediumOutputTests(unittest.TestCase):
    def test_streaming_grounding_preserves_exact_substrings_and_budget(self):
        state = State(messages=[], label=Label.NORMAL, section=TOP1, candidate_answer="기간은 12일로 고정돼요.")
        stream = AsyncMock(return_value='{"unsupported":["12일로 고정돼요"]}')
        with patch("pipelines.kimjaehak.medium_output.stream_text", stream):
            asyncio.run(Pipeline(SEARCHER, FakeLLM({}))._check_output(state))
        self.assertEqual(state.unsupported, ["12일로 고정돼요"])
        self.assertEqual(state.errors, [])
        self.assertEqual(stream.call_args.kwargs["effort"], "medium")
        self.assertEqual(stream.call_args.kwargs["max_tokens"], 4000)

    def test_blocked_input_keeps_existing_output_rule_without_streaming(self):
        state = State(messages=[], label=Label.HARMFUL_REQUEST, candidate_answer="candidate")
        stream = AsyncMock()
        with patch("pipelines.kimjaehak.medium_output.stream_text", stream):
            asyncio.run(Pipeline(SEARCHER, FakeLLM({}))._check_output(state))
        stream.assert_not_called()
        self.assertFalse(state.safe)


if __name__ == "__main__":
    unittest.main()
