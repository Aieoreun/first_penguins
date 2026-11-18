"""Guard-only category gate threshold; shared fitted models stay immutable."""
from copy import copy

from .peace_guard_scheduling import FastParallelInput
from .peace_span_variants import CurrentSpan80


class GuardGateThreshold:
    guard_gate_threshold = .90

    async def _classify(self, state):
        # Use the independent endpoint marker, not candidate presence or debug:
        # parent classifiers can replace debug while handling the request.
        if getattr(state, '_material_budget_guard', None) is not True:
            return await super()._classify(state)
        request_pipeline = copy(self)
        request_pipeline.learned_gate_threshold = self.guard_gate_threshold
        # Keep the C3000 -> C1000/subtype chain and its label mapping intact.
        # The subtype confidence threshold and all fitted model objects remain
        # inherited/shared; only this request's category threshold differs.
        return await super(GuardGateThreshold, request_pipeline)._classify(state)


class CurrentGate90(GuardGateThreshold, CurrentSpan80):
    pass


class FastGate90(GuardGateThreshold, FastParallelInput):
    pass
