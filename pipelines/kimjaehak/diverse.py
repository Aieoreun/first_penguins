"""Vote across two requested-operation checks and one original boundary check."""
from pipelines.v1.pipeline import CLASSIFY_EFFORT, CLASSIFY_TIMEOUT
from pipelines.v2.pipeline import Pipeline as V2
from .intent import BLOCK_SYSTEM
from .voted import Pipeline as Voted


class Pipeline(Voted):
    async def _block_check(self, state, check, index):
        if index == 0:
            return await super()._block_check(state, check, index)
        return await V2._ask(self, state, "block", BLOCK_SYSTEM, check, CLASSIFY_EFFORT, CLASSIFY_TIMEOUT)
