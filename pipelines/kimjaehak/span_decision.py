"""Let extremely confident scratch span predictions resolve grounding too."""
from core.schema import Decision
from .guard_fitted import Pipeline as GuardHybrid


class Pipeline(GuardHybrid):
    span_decision_threshold = .99

    async def _check_output(self, state):
        await super()._check_output(state)
        if state.decision is not Decision.ALLOW:
            return
        spans, confidence = self.span_tagger.predict(state.candidate_answer or "")
        if confidence >= self.span_decision_threshold:
            # This experiment explicitly changes grounding, unlike the .95
            # overlap-only boundary refinement. Input and safety stay intact.
            state.unsupported = spans
            state.grounded = not spans
            state.debug["learned_span_decision"] = True
