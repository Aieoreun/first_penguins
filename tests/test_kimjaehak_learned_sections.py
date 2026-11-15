import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from core.state import Chunk, State
from pipelines.kimjaehak.learned_sections import Pipeline, WholeKB
from tests.test_kimjaehak_learned_gate import Probability


class LearnedSectionTests(unittest.TestCase):
    def run_section(self, cls, section, confidence):
        pipe = object.__new__(cls)
        pipe._known_sections = {"FIRST", "SECOND"}
        pipe.section_vectorizer = SimpleNamespace(transform=Mock(return_value="features"))
        pipe.section_classifier = SimpleNamespace(classes_=[section], predict_proba=Mock(return_value=[Probability([confidence])]))
        state = State(messages=[],section="FIRST",chunks=[Chunk("FIRST","text")])
        with patch("pipelines.kimjaehak.learned_gate.Pipeline._classify", AsyncMock()):
            asyncio.run(pipe._classify(state))
        return state

    def test_retrieved_candidate_rule_and_whole_kb_rule_are_distinct(self):
        self.assertEqual(self.run_section(Pipeline,"SECOND",.999).section,"FIRST")
        self.assertEqual(self.run_section(WholeKB,"SECOND",.999).section,"SECOND")

    def test_low_confidence_or_unknown_sections_preserve_search(self):
        self.assertEqual(self.run_section(WholeKB,"SECOND",.9).section,"FIRST")
        self.assertEqual(self.run_section(WholeKB,"UNKNOWN",.999).section,"FIRST")


if __name__ == "__main__":
    unittest.main()
