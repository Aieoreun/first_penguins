"""Skip unused query embeddings on already-trusted tutor input/source routes."""
from copy import copy

from .material_budget import Pipeline as MaterialBudget


class _WithoutEmbedding:
    can_embed = False

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)


class Pipeline(MaterialBudget):
    tutor_skip_trusted_embedding = True

    def _skip_query_embedding(self, state):
        if (getattr(state, '_material_budget_guard', True)
                or not self.searcher.has_embeddings or not self.llm.can_embed
                or self.learned_section_require_retrieved):
            return False
        probability = self.classifier.predict_proba(self.vectorizer.transform([state.last_user]))[0]
        index = probability.argmax()
        if (str(self.classifier.classes_[index]) not in {'normal','pii_disclosure'}
                or probability[index] < self.learned_gate_threshold):
            return False
        sections = self.section_classifier.predict_proba(
            self.section_vectorizer.transform([state.last_user]))[0]
        index = sections.argmax()
        return bool(sections[index] >= self.tutor_section_threshold
                    and str(self.section_classifier.classes_[index]) in self._known_sections)

    async def _search(self, state, extra=''):
        if not self._skip_query_embedding(state):
            return await super()._search(state, extra)
        current = copy(self)
        current.llm = _WithoutEmbedding(self.llm)
        await super(Pipeline, current)._search(state, extra)
        state.debug['trusted_embedding_skipped'] = True
        # The unchanged classifier may replace debug. No routing decision is
        # inferred from this diagnostic marker and no model/LLM is mutated.
