import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.candidate import Pipeline as Candidate
from pipelines.kimjaehak.multiground import Pipeline as MultiGround
from pipelines.kimjaehak.uncertain_sources import Pipeline


class UncertainSourcesTests(unittest.TestCase):
    def test_routing_preserves_high_confidence_and_policy_paths(self):
        pipe = Pipeline.__new__(Pipeline)
        state = State(messages=[], label=Label.NORMAL,
                      chunks=[Chunk("A", "first"), Chunk("B", "second")])
        with patch.object(MultiGround, "_generate", new_callable=AsyncMock) as multi, \
             patch.object(Candidate, "_generate", new_callable=AsyncMock) as single:
            asyncio.run(pipe._generate(state))
            multi.assert_awaited_once_with(pipe, state)
            single.assert_not_awaited()
            state.debug["learned_section"] = True
            asyncio.run(pipe._generate(state))
            state.debug.clear()
            state.label = Label.OUT_OF_MATERIAL
            asyncio.run(pipe._generate(state))
            self.assertEqual(single.await_count, 2)
            self.assertEqual(multi.await_count, 1)


if __name__ == "__main__":
    unittest.main()
