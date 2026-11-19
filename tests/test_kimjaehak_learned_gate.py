import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.learned_gate import Pipeline
from tests.test_pipeline_v2 import SEARCHER


class Probability(list):
    def argmax(self):
        return self.index(max(self))


class LearnedGateTests(unittest.TestCase):
    def pipeline(self, category, probability):
        vectorizer = SimpleNamespace(transform=Mock(return_value="features"))
        classifier = SimpleNamespace(classes_=[category], predict_proba=Mock(return_value=[Probability([probability])]))
        with patch("pipelines.kimjaehak.learned_gate.train_public_classifier", return_value=(vectorizer,classifier)) as fit:
            pipe = Pipeline(SEARCHER, None)
        fit.assert_called_once_with(1000)
        return pipe

    def test_confident_prediction_uses_features_and_no_request_identifier(self):
        pipe = self.pipeline("normal", .999)
        state = State(messages=[{"role":"user","content":"개념을 설명해 줘"}])
        asyncio.run(pipe._classify(state))
        self.assertEqual(state.label, Label.NORMAL)
        pipe.vectorizer.transform.assert_called_once_with(["개념을 설명해 줘"])

    def test_uncertain_or_ambiguous_subtype_keeps_llm_path(self):
        for category, probability in (("normal", .99), ("harmful_content", .999), ("invalid", .999)):
            pipe = self.pipeline(category, probability)
            with patch("pipelines.kimjaehak.voted.Pipeline._classify", AsyncMock()) as fallback:
                state = State(messages=[])
                asyncio.run(pipe._classify(state))
            fallback.assert_awaited_once_with(state)


if __name__ == "__main__":
    unittest.main()
