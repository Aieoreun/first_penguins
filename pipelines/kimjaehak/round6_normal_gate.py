"""A slightly broader public-trained normal gate, isolated to tutor requests."""
from copy import copy

from .material_budget import Pipeline as MaterialBudget


class Pipeline(MaterialBudget):
    tutor_normal_gate_threshold = .94

    async def _classify(self, state):
        current = self
        if not getattr(state, '_material_budget_guard', True):
            probabilities = self.classifier.predict_proba(
                self.vectorizer.transform([state.last_user]))[0]
            index = probabilities.argmax()
            if (str(self.classifier.classes_[index]) == 'normal'
                    and self.tutor_normal_gate_threshold <= probabilities[index] < self.learned_gate_threshold):
                current = copy(self)
                current.learned_gate_threshold = self.tutor_normal_gate_threshold
        # Retain all existing classification priority and learned-source
        # postprocessing; never set a label directly or mutate the shared model.
        await super(Pipeline, current)._classify(state)
        if current is not self and state.debug.get('learned_gate'):
            state.debug['tutor_normal_gate_threshold'] = self.tutor_normal_gate_threshold
