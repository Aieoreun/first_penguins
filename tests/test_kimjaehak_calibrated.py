import asyncio
import unittest

from core.schema import Label
from pipelines.kimjaehak.calibrated import Pipeline, ResponseState, low_signal, FIXED
from tests.test_pipeline import FakeLLM
from tests.test_pipeline_v2 import SEARCHER


class CalibratedTests(unittest.TestCase):
    def test_gibberish_overrides_earlier_course_question_without_api_calls(self):
        llm = FakeLLM({})
        state = asyncio.run(Pipeline(SEARCHER, llm).tutor({"messages": [
            {"role": "user", "content": "파이썬 함수는 왜 만들어요?"},
            {"role": "assistant", "content": "동작을 묶어 쓸 수 있어요."},
            {"role": "user", "content": "ㅠㅠㅠ"},
        ]}))
        self.assertEqual(state.to_tutor_response()["action"], "clarify")
        self.assertEqual(llm.calls, [])

    def test_short_meaningful_answers_are_not_overridden(self):
        for text in ("ㅇㅇ", "ㄱㄱ", "네", "4", "아니요", "이해가 안 돼요", "print(42)"):
            self.assertFalse(low_signal(text), text)
        for text in ("ㅋㅋㅋ", "ㅁㄴㅇㄹ", "...", "??"):
            self.assertTrue(low_signal(text), text)

    def test_clear_refusal_preserves_action_and_other_state(self):
        state = ResponseState(messages=[], label=Label.HARMFUL_SPEECH)
        response = state.to_tutor_response()
        self.assertEqual(response["answer"], FIXED[Label.HARMFUL_SPEECH])
        self.assertEqual(response["action"], "redirect")
        self.assertEqual(response["citations"], [])


if __name__ == "__main__":
    unittest.main()
