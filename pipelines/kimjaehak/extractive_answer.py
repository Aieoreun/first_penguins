"""Return runtime-selected organizer KB passages without inventing facts."""
from core.schema import ANSWER_MAX_CHARS, Label
from .guard_fitted import ContextFast


class Section(ContextFast):
    tutor_extract_scope = 'section'

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._extract_guard = guard
        return state

    def _excerpt(self, state):
        return self.searcher.section_text(state.section)

    async def _generate(self, state):
        if (getattr(state, '_extract_guard', False) or not state.section
                or state.label not in {Label.NORMAL, Label.PII_DISCLOSURE}):
            return await super()._generate(state)
        excerpt = self._excerpt(state).strip()
        answer = '관련 교재 내용을 인용할게요.\n\n' + excerpt
        if not excerpt or len(answer) > ANSWER_MAX_CHARS:
            return await super()._generate(state)
        state.answer, state.citations = answer, [state.section]


class Chunk(Section):
    tutor_extract_scope = 'chunk'

    def _excerpt(self, state):
        query = '\n'.join(message['content'] for message in state.messages)
        scores = self.searcher.bm25.scores(query)
        choices = [(score, chunk.text) for chunk, score in zip(self.searcher.chunks, scores)
                   if chunk.section_id == state.section]
        if not choices:
            return ''
        return max(choices, key=lambda row: row[0])[1]
