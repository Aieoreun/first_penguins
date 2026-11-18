"""Keep ContextFast safety routing while retaining fusion grounding scheduling."""
from core.schema import Decision
from .peace_guard_fusion import Pipeline as Fusion
from .peace_guard_variants import RequestedAction
from .peace_ground_variants import ProvenanceGrounding, RelationGrounding


class RoutedSafety(Fusion):
    guard_safety_after_input = True
    guard_independent_safety = False
    guard_routed_safety = True
    guard_safety_timeout = Fusion.refusal_timeout
    guard_safety_system = Fusion.policy_system

    async def _fusion_safety(self, state):
        if state.decision is not Decision.BLOCK:
            state.safe = True
            return
        # The inherited ContextFast BLOCK path sets safe=False, skips grounding,
        # and runs BroadReview only when the existing refusal pattern matches.
        # Reuse its prompt/hints, strict bool parser, 8s bound and error handling.
        await super()._check_output(state)


class RoutedRequested(RoutedSafety, RequestedAction):
    pass


class RoutedRelation(RoutedSafety):
    unsupported_system = RelationGrounding.unsupported_system


class RoutedProvenance(RoutedSafety):
    unsupported_system = ProvenanceGrounding.unsupported_system
