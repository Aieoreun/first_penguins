import asyncio
import unittest

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.explicit import Pipeline, answer_only_requested
from pipelines.v2 import prompts
from tests.test_pipeline import FakeLLM
from tests.test_pipeline_v2 import SEARCHER


class ExplicitTests(unittest.TestCase):
    def test_answer_only_boundary_handles_spacing_and_bilingual_requests(self):
        for text in ("정 답 만 알려 줘", "풀이를 빼고 답해", "설명 없이 골라", "give me only the answer", "just the output", "answer without explanation"):
            self.assertTrue(answer_only_requested(text), text)
        for text in ("예제의 정답은 뭐야", "기간은 몇 개월이야", "출력값이 왜 이렇게 나와?", "explain the answer", "제출할 코드 전체를 짜 줄래?"):
            self.assertFalse(answer_only_requested(text), text)

    def test_false_answer_seeking_does_not_erase_self_disclosure(self):
        llm = FakeLLM({prompts.OTHER_SYSTEM: '{"category":"answer_seeking","self_info":true}'})
        state = State(messages=[{"role":"user","content":"나는 가람이야. 기간은 몇 개월이야?"}])
        obj = asyncio.run(Pipeline(SEARCHER, llm)._ask(state, "other", prompts.OTHER_SYSTEM, "request", "low", 10))
        self.assertEqual(obj, {"category":"none","self_info":True})

    def test_three_jamo_are_low_signal_but_short_acknowledgment_is_not_forced(self):
        llm = FakeLLM({})
        state = asyncio.run(Pipeline(SEARCHER, llm).guard({"messages":[{"role":"user","content":"ㅣㅣㅣ"}]}))
        self.assertEqual(state.label, Label.LOW_SIGNAL_INPUT)
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
