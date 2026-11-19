import asyncio
import time
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.context_grounding import Pipeline as ContextGrounding
from pipelines.kimjaehak.hybrid_output import Pipeline


class HybridOutputTests(unittest.TestCase):
    def pipe_state(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe.llm = None
        state = State(messages=[],candidate_answer="잘못된 값이에요.",label=Label.NORMAL)
        state._ground_deadline = time.monotonic()+.03
        return pipe,state

    def test_reasoning_timeout_keeps_completed_fast_result(self):
        pipe,state = self.pipe_state()
        async def fast(self, low):
            low.unsupported=["잘못된 값"]
            low.grounded=False
        async def slow(*args, **kwargs):
            await asyncio.sleep(1)
        with patch.object(ContextGrounding,"_check_output",fast), \
             patch("pipelines.kimjaehak.hybrid_output.stream_text",side_effect=slow):
            started=time.monotonic();asyncio.run(pipe._check_output(state))
        self.assertLess(time.monotonic()-started,.3)
        self.assertEqual(state.unsupported,["잘못된 값"])
        self.assertFalse(state.grounded)
        self.assertFalse(state.debug["hybrid_reasoning_used"])
        self.assertIn("hybrid_reasoning:TimeoutError",state.errors)

    def test_successful_reasoning_can_correct_fast_false_positive(self):
        pipe,state = self.pipe_state()
        async def fast(self, low):
            low.unsupported=["잘못된 값"]
        with patch.object(ContextGrounding,"_check_output",fast), \
             patch("pipelines.kimjaehak.hybrid_output.stream_text",new_callable=AsyncMock,return_value='{"unsupported":[]}'):
            asyncio.run(pipe._check_output(state))
        self.assertEqual(state.unsupported,[])
        self.assertTrue(state.grounded)
        self.assertTrue(state.debug["hybrid_reasoning_used"])


if __name__=="__main__":unittest.main()
