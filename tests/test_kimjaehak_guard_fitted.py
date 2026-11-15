import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from pipelines.kimjaehak.guard_fitted import Pipeline
from pipelines.kimjaehak.refusal_review import BroadPipeline
from tests.test_kimjaehak_candidate import classifier


class GuardFittedTests(unittest.TestCase):
    def test_endpoint_model_does_not_change_tutor_or_harmful_subtypes(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe.guard_vectorizer, pipe.guard_classifier = classifier("normal")
        body = {"messages": [{"role":"user", "content":"학습 질문"}]}
        with patch.object(BroadPipeline, "_classify", new_callable=AsyncMock) as inherited:
            for with_candidate in (False, True):
                state = pipe._new_state({**body, **({"candidate_answer":"후보"} if with_candidate else {})}, True)
                asyncio.run(pipe._classify(state))
                self.assertEqual(state.label, Label.NORMAL)
                self.assertTrue(state.debug["learned_guard_only"])
            inherited.assert_not_awaited()
            tutor = pipe._new_state(body, False)
            asyncio.run(pipe._classify(tutor))
            inherited.assert_awaited_once_with(tutor)
            inherited.reset_mock()
            pipe.guard_vectorizer, pipe.guard_classifier = classifier("harmful_content")
            guard = pipe._new_state(body, True)
            asyncio.run(pipe._classify(guard))
            inherited.assert_awaited_once_with(guard)


if __name__ == "__main__":
    unittest.main()
