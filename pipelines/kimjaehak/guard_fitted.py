"""Increase scratch category fitting only for the guard endpoint."""
from core.schema import parse_label
from .learned_gate import train_public_classifier
from .refusal_review import BroadPipeline, BroadReview
from .learned_output import Consensus


class GuardFitting:
    guard_only_c = 3000

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self.guard_vectorizer, self.guard_classifier = train_public_classifier(self.guard_only_c)

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state.debug["guard_endpoint"] = guard
        return state

    async def _classify(self, state):
        if state.debug.get("guard_endpoint"):
            # Keep the models immutable and separate: no shared-model swapping
            # while concurrent tutor requests use the original C=1000 path.
            probabilities = self.guard_classifier.predict_proba(
                self.guard_vectorizer.transform([state.last_user]))[0]
            index = probabilities.argmax()
            category = str(self.guard_classifier.classes_[index])
            label = parse_label(category)
            if (probabilities[index] >= self.learned_gate_threshold
                    and label is not None and category != "harmful_content"):
                state.label = label
                state.section = state.chunks[0].section_id if state.chunks else None
                state.debug = {"learned_guard_only": True, "confidence": float(probabilities[index])}
                return
        # Harmful subtype and every tutor request preserve the original models.
        await super()._classify(state)


class Pipeline(GuardFitting, BroadPipeline):
    pass


class Fast(GuardFitting, BroadReview, Consensus):
    """Same endpoint-specific input and safety checks, with low-only grounding."""
    pass
