import asyncio
import json
import unittest

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.evidence_answer import Pipeline


class EvidenceAnswerTests(unittest.TestCase):
    def run_answer(self, quote):
        class Searcher:
            def section_text(self, section):
                return "가상 도서관에서는 한 번에 세 권을 빌려요."

        class LLM:
            calls = 0
            async def chat(self, system, user, **kwargs):
                self.calls += 1
                if kwargs.get("json_mode"):
                    return json.dumps({"quotes": quote, "answer": "한 번에 세 권을 빌려요."})
                return "자료의 대출 권수를 확인해요."

        pipe = object.__new__(Pipeline)
        pipe.searcher, pipe.llm = Searcher(), LLM()
        state = State(messages=[], label=Label.NORMAL, section="TEST-01")
        asyncio.run(pipe._generate(state))
        return pipe, state

    def test_exact_excerpt_is_accepted_without_exposing_intermediate_quotes(self):
        pipe, state = self.run_answer(["한 번에 세 권을 빌려요."])
        self.assertEqual(pipe.llm.calls, 1)
        self.assertEqual(state.to_tutor_response()["answer"], "한 번에 세 권을 빌려요. (TEST-01)")
        self.assertEqual(state.errors, [])

    def test_invented_excerpt_falls_back(self):
        pipe, state = self.run_answer(["한 번에 다섯 권을 빌려요."])
        self.assertEqual(pipe.llm.calls, 2)
        self.assertEqual(state.errors, ["evidence_answer:ValueError"])
        self.assertIn("대출 권수를 확인", state.answer)

    def test_missing_excerpt_array_falls_back(self):
        pipe, state = self.run_answer(None)
        self.assertEqual(pipe.llm.calls, 2)
        self.assertEqual(state.errors, ["evidence_answer:ValueError"])


if __name__ == "__main__":
    unittest.main()
