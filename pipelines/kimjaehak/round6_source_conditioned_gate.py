"""Recheck trusted normal input when its learned source is not trusted."""
from copy import copy

from .material_budget import Pipeline as MaterialBudget


class Pipeline(MaterialBudget):
    tutor_source_conditioned_gate = True
    source_conditioned_input_threshold = .95
    source_conditioned_section_threshold = .90
    source_conditioned_fallback_threshold = 1.01

    async def _classify(self, state):
        current = self
        if not getattr(state, '_material_budget_guard', True):
            probabilities = self.classifier.predict_proba(
                self.vectorizer.transform([state.last_user]))[0]
            index = probabilities.argmax()
            if (str(self.classifier.classes_[index]) == 'normal'
                    and probabilities[index] >= self.source_conditioned_input_threshold):
                sections = self.section_classifier.predict_proba(
                    self.section_vectorizer.transform([state.last_user]))[0]
                section_index = sections.argmax()
                section = str(self.section_classifier.classes_[section_index])
                if (sections[section_index] < self.source_conditioned_section_threshold
                        or section not in self._known_sections):
                    current = copy(self)
                    current.learned_gate_threshold = self.source_conditioned_fallback_threshold
        # Reuse the complete original policy/material checks and section gates.
        # No label is forced and the shared classifier or threshold is unchanged.
        await super(Pipeline, current)._classify(state)
        if current is not self:
            state.debug['source_conditioned_gate_rechecked'] = True
