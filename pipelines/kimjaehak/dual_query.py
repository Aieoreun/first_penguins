"""Give uncertain tutors the best full-dialogue and last-question sections."""
import copy

from core.schema import Label
from core.state import Chunk
from .guard_fitted import ContextFast
from .round2 import Sources


class Pipeline(Sources):
    answer_sections = 2
    tutor_dual_query = True

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._dual_guard = guard
        state._dual_sections = []
        return state

    async def _search(self, state, extra=''):
        await super()._search(state, extra)
        if getattr(state, '_dual_guard', False) or not state.history or not state.chunks:
            return
        last = self.searcher.search(state.last_user, bm25_top=1, embedding_top=0)
        if not last or last[0].section_id == state.chunks[0].section_id:
            return
        current, alternative = state.chunks[0], last[0]
        state._dual_sections = [current.section_id, alternative.section_id]
        # Add only one small lexical result. Preserve all original dense/BM25
        # candidates for the material check, without another embedding call.
        key = (alternative.section_id, alternative.text)
        state.chunks = [current, Chunk(*key)] + [c for c in state.chunks[1:]
                                               if (c.section_id, c.text) != key]

    async def _generate(self, state):
        ids = getattr(state, '_dual_sections', [])
        if (state.label not in {Label.NORMAL, Label.PII_DISCLOSURE} or not state.section
                or len(ids) != 2 or state.debug.get('learned_section')):
            return await ContextFast._generate(self, state)
        alternative = next(sid for sid in ids if sid != state.section)
        selected = copy.copy(state)
        selected.chunks = [Chunk(alternative, self.searcher.section_text(alternative))]
        selected.errors, selected.debug = [], dict(state.debug)
        await super()._generate(selected)
        state.answer, state.citations, state.section = selected.answer, list(selected.citations), selected.section
        state.errors.extend(selected.errors)

