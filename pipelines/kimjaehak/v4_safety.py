"""Retain v4 grounding; replace only safety with the explicit Peace-style check."""
import copy

from pipelines.v1.pipeline import Pipeline as V1
from pipelines.v4.pipeline import Pipeline as V4
from .precise_output import PreciseOutput


class OriginalGround(PreciseOutput):
    async def _ground(self, state):
        # Isolate the original check's safe=True assignment from concurrent safety.
        shadow = copy.copy(state)
        shadow.errors = []
        await V1._check_output(self, shadow)
        state.errors.extend(shadow.errors)
        return shadow.unsupported


class Pipeline(V4, OriginalGround):
    pass
