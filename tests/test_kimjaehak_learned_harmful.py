import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.learned_harmful import Pipeline
from tests.test_kimjaehak_learned_gate import Probability


class HarmfulGateTests(unittest.TestCase):
    def pipeline(self, category="harmful_content", confidence=.99, action="redirect", sub_confidence=.99):
        pipe = object.__new__(Pipeline)
        pipe.vectorizer = pipe.harmful_vectorizer = SimpleNamespace(transform=Mock(return_value="features"))
        pipe.classifier = SimpleNamespace(classes_=[category], predict_proba=Mock(return_value=[Probability([confidence])]))
        pipe.harmful_classifier = SimpleNamespace(classes_=[action], predict_proba=Mock(return_value=[Probability([sub_confidence])]))
        return pipe

    def test_confident_abuse_and_request_have_different_tutor_actions(self):
        for action, label in (("redirect", Label.HARMFUL_SPEECH), ("refuse", Label.HARMFUL_REQUEST)):
            pipe = self.pipeline(action=action)
            state = State(messages=[{"role":"user","content":"분류할 문장"}])
            asyncio.run(pipe._classify(state))
            self.assertEqual(state.label, label)
            self.assertEqual(state.to_tutor_response()["action"], action)
            self.assertEqual(state.to_guard_response()["input"], {"decision":"block","category":"harmful_content"})

    def test_both_confidences_are_required_and_other_categories_fall_back(self):
        for kwargs in ({"confidence":.94}, {"sub_confidence":.94}, {"category":"normal"}, {"action":"unknown"}):
            pipe = self.pipeline(**kwargs)
            state = State(messages=[])
            with patch("pipelines.kimjaehak.learned_sections.Pipeline._classify", AsyncMock()) as fallback:
                asyncio.run(pipe._classify(state))
            fallback.assert_awaited_once_with(state)


if __name__ == "__main__":
    unittest.main()
