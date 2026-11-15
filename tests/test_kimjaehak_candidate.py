import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.candidate import Pipeline
from pipelines.kimjaehak.concise_answer import ANSWER_SYSTEM
from pipelines.kimjaehak.plain_abstain import ABSTAIN
from tests.test_kimjaehak_learned_gate import Probability
from tests.test_pipeline_v2 import SEARCHER


def classifier(label):
    return (SimpleNamespace(transform=Mock(return_value="features")),
            SimpleNamespace(classes_=[label], predict_proba=Mock(return_value=[Probability([.999])])) )


class CandidateIntegrationTests(unittest.TestCase):
    def test_composition_initializes_all_models_and_preserves_all_four_changes(self):
        section = SEARCHER.section_ids()[0]
        class LLM:
            calls = []
            async def chat(self, system, user, **kwargs):
                self.calls.append((system, user))
                return '{"unsupported": []}' if kwargs.get("json_mode") else "자료의 조건을 확인해요."
        with patch("pipelines.kimjaehak.learned_gate.train_public_classifier", return_value=classifier("normal")) as category, \
             patch("pipelines.kimjaehak.learned_sections.train_section_classifier", return_value=classifier(section)) as evidence, \
             patch("pipelines.kimjaehak.learned_harmful.train_harmful_classifier", return_value=classifier("redirect")) as harmful:
            pipe = Pipeline(SEARCHER, LLM())
        for fit in (category, evidence, harmful):
            fit.assert_called_once_with(1000)
        state = State(messages=[{"role":"user","content":"조건을 알려 주세요"}], chunks=[Chunk(section,"자료")])
        asyncio.run(pipe._classify(state))
        self.assertEqual(state.label, Label.NORMAL)
        asyncio.run(pipe._generate(state))
        self.assertEqual(pipe.llm.calls[-1][0], ANSWER_SYSTEM)
        before = len(pipe.llm.calls)
        state.label = Label.OUT_OF_MATERIAL
        asyncio.run(pipe._generate(state))
        self.assertEqual(state.answer, ABSTAIN)
        self.assertEqual(state.citations, [])
        self.assertEqual(len(pipe.llm.calls), before)
        state.label, state.candidate_answer = Label.NORMAL, "자료의 조건을 확인해요."
        asyncio.run(pipe._check_output(state))
        payload = json.loads(pipe.llm.calls[-1][1])
        self.assertEqual(payload["대화"], state.messages)
        self.assertEqual(len(payload["교재"]), 1)
        pipe.classifier = classifier("harmful_content")[1]
        asyncio.run(pipe._classify(state))
        self.assertEqual(state.label, Label.HARMFUL_SPEECH)


if __name__ == "__main__":
    unittest.main()
