import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.context_policy import Pipeline as ContextPolicy, PolicyState
from pipelines.kimjaehak.learned_output import Pipeline, Consensus
from pipelines.kimjaehak.voted_output import Pipeline as VotedOutput


class LearnedOutputTests(unittest.TestCase):
    def test_refinement_preserves_grounding_safety_and_input_decisions(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe.span_tagger = SimpleNamespace(predict=Mock(return_value=(["12일이에요"], .99)))
        state = State(messages=[], candidate_answer="기간은 12일이에요.", label=Label.NORMAL)
        async def check(self, state):
            state.unsupported = ["기간은 12일이에요"]
            state.grounded, state.safe = False, True
        with patch.object(ContextPolicy, "_check_output", check):
            asyncio.run(pipe._check_output(state))
        self.assertEqual(state.unsupported, ["12일이에요"])
        self.assertEqual((state.grounded, state.safe, state.label), (False, True, Label.NORMAL))
        self.assertTrue(state.debug["learned_span"])

    def test_consensus_composition_keeps_refinement_and_contextual_refusals(self):
        self.assertEqual(Consensus.output_votes, 5)
        self.assertIs(Consensus._check_output, Pipeline._check_output)
        self.assertIs(Consensus.state_class, PolicyState)
        self.assertIs(super(Pipeline, Consensus.__new__(Consensus))._check_output.__func__, VotedOutput._check_output)


if __name__ == "__main__":
    unittest.main()
