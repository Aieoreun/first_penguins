import asyncio
import unittest
from unittest.mock import AsyncMock

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.refusal_review import RefusalReview


class Base:
    async def _check_output(self, state):
        state.safe = False
        state.grounded = True
        state.unsupported = []


class Stub(RefusalReview, Base):
    pass


class RefusalReviewTests(unittest.TestCase):
    def check(self, candidate, verdict, label=Label.HARMFUL_REQUEST):
        pipe = Stub()
        pipe.llm = type("LLM", (), {"chat": AsyncMock(return_value=verdict)})()
        state = State(messages=[{"role":"user", "content":"위험한 요청"}],
                      candidate_answer=candidate, label=label)
        asyncio.run(pipe._check_output(state))
        return pipe, state

    def test_safe_refusal_can_pass_with_blocked_input(self):
        pipe, state = self.check("그 요청은 도울 수 없어요. 예방 방법을 배워봐요.", '{"safe":true}')
        self.assertTrue(state.safe)
        self.assertTrue(state.grounded)
        self.assertEqual(state.label, Label.HARMFUL_REQUEST)
        pipe.llm.chat.assert_awaited_once()

    def test_refusal_word_does_not_override_unsafe_verdict(self):
        _, state = self.check("도울 수 없어요. 하지만 요청한 행동을 해드릴게요.", '{"safe":false}')
        self.assertFalse(state.safe)

    def test_invalid_verdict_preserves_previous_output(self):
        _, state = self.check("도울 수 없어요.", '{"safe":"true"}')
        self.assertFalse(state.safe)
        self.assertIn("refusal_review:ValueError", state.errors)

    def test_other_candidates_and_allowed_inputs_skip_review(self):
        for candidate, label in (("네, 해드릴게요.", Label.HARMFUL_REQUEST),
                                 ("도울 수 없어요.", Label.NORMAL)):
            pipe, _ = self.check(candidate, '{"safe":true}', label)
            pipe.llm.chat.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
