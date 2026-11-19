"""Reuse a verified source already selected by the tutor material check."""
from core.schema import Label
from .guard_fitted import ContextFast


class Pipeline(ContextFast):
    tutor_material_source = True
    material_quote_min_chars = 8

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._material_source_guard = guard
        state._material_source_result = None
        return state

    async def _ask(self, state, name, system, user, effort, timeout):
        result = await super()._ask(state, name, system, user, effort, timeout)
        if name == 'material' and not getattr(state, '_material_source_guard', False):
            # Request-local prediction only; never log it or reuse other requests.
            state._material_source_result = result
        return result

    async def _classify(self, state):
        await super()._classify(state)
        if (getattr(state, '_material_source_guard', False) or state.debug.get('learned_section')
                or state.label not in {Label.NORMAL, Label.PII_DISCLOSURE, Label.ANSWER_SEEKING}):
            return
        result = getattr(state, '_material_source_result', None) or {}
        section, quote = result.get('section'), result.get('quote')
        if (not isinstance(section, str) or section not in {c.section_id for c in state.chunks}
                or not isinstance(quote, str)):
            return
        compact = ''.join(quote.split())
        if (len(compact) < self.material_quote_min_chars
                or compact not in ''.join(self.searcher.section_text(section).split())):
            return
        state.section = section
        state.debug['tutor_material_source'] = True
