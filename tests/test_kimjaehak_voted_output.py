import asyncio
import unittest
from unittest.mock import patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.context_output import Pipeline as ContextOutput
from pipelines.kimjaehak.voted_output import Pipeline, consensus_index


class VotedOutputTests(unittest.TestCase):
    def test_majority_grounded_beats_single_long_error_span(self):
        self.assertEqual(consensus_index("정상 설명입니다.", [["정상 설명입니다."], [], []]), 1)

    def test_selects_real_verdict_between_short_and_long_spans(self):
        candidate = "대출은 12일로 고정돼 있어요. 다른 조건은 같아요."
        votes = [["12일로"], ["12일로 고정돼 있어요."], ["대출은 12일로 고정돼 있어요."]]
        self.assertEqual(consensus_index(candidate, votes), 1)

    def test_failed_calls_do_not_vote_for_grounded(self):
        pipe = Pipeline.__new__(Pipeline)
        state = State(messages=[], label=Label.NORMAL, candidate_answer="12일이에요.", errors=["prior:parse"])
        counter = 0
        async def check(self, trial):
            nonlocal counter
            number = counter
            counter += 1
            await asyncio.sleep(0)
            if number < 2:
                trial.errors.append("context_grounding:TimeoutError")
            else:
                trial.unsupported = ["12일이에요."]
                trial.grounded = False
        with patch.object(ContextOutput, "_check_output", check):
            asyncio.run(pipe._check_output(state))
        self.assertFalse(state.grounded)
        self.assertEqual(state.unsupported, ["12일이에요."])
        self.assertEqual(state.errors, ["prior:parse", "context_grounding:TimeoutError", "context_grounding:TimeoutError"])


if __name__ == "__main__":
    unittest.main()
