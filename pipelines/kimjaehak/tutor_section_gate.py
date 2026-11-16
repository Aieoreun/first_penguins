"""Tutor-only lower-confidence section fitting ablation, with fixed guard."""
from core.schema import Label
from .guard_fitted import ContextFast


class Pipeline(ContextFast):
    tutor_section_threshold = .90

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._section_gate_guard = guard
        return state

    async def _classify(self, state):
        await super()._classify(state)
        if (getattr(state, '_section_gate_guard', False) or state.debug.get('learned_section')
                or state.label not in {Label.NORMAL, Label.PII_DISCLOSURE, Label.ANSWER_SEEKING}):
            return
        probabilities = self.section_classifier.predict_proba(
            self.section_vectorizer.transform([state.last_user]))[0]
        index = probabilities.argmax()
        section = str(self.section_classifier.classes_[index])
        if probabilities[index] >= self.tutor_section_threshold and section in self._known_sections:
            state.section = section
            state.debug['tutor_section_gate'] = float(probabilities[index])
