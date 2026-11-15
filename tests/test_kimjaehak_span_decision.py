import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.guard_fitted import Pipeline as GuardHybrid
from pipelines.kimjaehak.span_decision import Pipeline


class SpanDecisionTests(unittest.TestCase):
    def check(self, spans, confidence, label=Label.NORMAL):
        pipe = Pipeline.__new__(Pipeline)
        pipe.span_tagger = SimpleNamespace(predict=Mock(return_value=(spans, confidence)))
        state = State(messages=[], candidate_answer="후보 답변", label=label)
        async def original(self, state):
            state.unsupported = ["후보"]
            state.grounded, state.safe = False, False
        with patch.object(GuardHybrid, "_check_output", original):
            asyncio.run(pipe._check_output(state))
        return pipe, state

    def test_confident_grounding_can_change_without_changing_safety(self):
        _, state = self.check([], .999)
        self.assertEqual(state.unsupported, [])
        self.assertTrue(state.grounded)
        self.assertFalse(state.safe)
        self.assertEqual(state.label, Label.NORMAL)

    def test_lower_confidence_preserves_original_verdict(self):
        _, state = self.check([], .98)
        self.assertEqual(state.unsupported, ["후보"])
        self.assertFalse(state.grounded)

    def test_blocked_candidate_does_not_use_grounding_override(self):
        pipe, state = self.check([], .999, Label.HARMFUL_REQUEST)
        pipe.span_tagger.predict.assert_not_called()
        self.assertFalse(state.grounded)


if __name__ == "__main__":
    unittest.main()
