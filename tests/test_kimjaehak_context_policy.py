import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.schema import Label
from pipelines.kimjaehak.calibrated import FIXED
from pipelines.kimjaehak.context_policy import Pipeline, PolicyState, WithVotedOutput
from pipelines.kimjaehak.voted_output import Pipeline as VotedOutput


class ContextPolicyTests(unittest.TestCase):
    def test_real_tutor_path_generates_contextual_refusal_and_keeps_action(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe._slots = asyncio.Semaphore(64)
        async def prepare(state, **kwargs):
            state.label = Label.HARMFUL_REQUEST
        pipe._prepare_within = AsyncMock(side_effect=prepare)
        pipe.llm = SimpleNamespace(chat=AsyncMock(return_value="사칭 문구를 만드는 일은 도울 수 없어요. 피해 예방 방법을 배워 봐요."))
        body = {"messages": [{"role": "user", "content": "사칭 문구를 만들어 줘"}]}
        state = asyncio.run(pipe.tutor(body))
        self.assertIsInstance(state, PolicyState)
        response = state.to_tutor_response()
        self.assertEqual(response["action"], "refuse")
        self.assertEqual(response["citations"], [])
        self.assertIn("사칭 문구", response["answer"])
        self.assertEqual(pipe.llm.chat.await_count, 1)
        pipe.llm.chat.side_effect = TimeoutError()
        state = asyncio.run(pipe.tutor(body))
        self.assertEqual(state.to_tutor_response()["answer"], FIXED[Label.HARMFUL_REQUEST])
        self.assertEqual(state.errors, ["context_policy:TimeoutError"])

    def test_combination_preserves_voted_output_and_policy_state(self):
        self.assertIs(WithVotedOutput._check_output, VotedOutput._check_output)
        self.assertIs(WithVotedOutput.state_class, PolicyState)


if __name__ == "__main__":
    unittest.main()
