import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from core.state import State
from pipelines.kimjaehak.calibrated import BLOCK_SYSTEM
from pipelines.kimjaehak.reasoned_tie import Pipeline
from tests.test_pipeline import FakeLLM
from tests.test_pipeline_v2 import SEARCHER


class ReasonedTieTests(unittest.TestCase):
    def test_agreement_does_not_call_medium(self):
        state = State(messages=[])
        pipe = Pipeline(SEARCHER, FakeLLM({BLOCK_SYSTEM:'{"category":"none"}'}))
        with patch("pipelines.kimjaehak.reasoned_tie.stream_text", AsyncMock()) as medium:
            answers = asyncio.run(pipe._vote_group(state, "block", "replaced", "request"))
        self.assertEqual(len(answers), 2)
        medium.assert_not_called()

    def test_tie_keeps_remaining_budget_and_failure_does_not_add_ballot(self):
        state = State(messages=[])
        pipe = Pipeline(SEARCHER, None)
        medium = AsyncMock(side_effect=TimeoutError())
        with patch("pipelines.kimjaehak.reasoned_tie.stream_text", medium):
            answer = asyncio.run(pipe._tie_vote(state, "block", "replaced", "request", 3.25))
        self.assertEqual(answer, {})
        self.assertEqual(state.errors, ["block_tie:TimeoutError"])
        self.assertEqual(medium.call_args.kwargs["timeout"], 3.25)
        self.assertEqual(medium.call_args.args[1], BLOCK_SYSTEM)


if __name__ == "__main__":
    unittest.main()
