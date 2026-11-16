"""Return runtime-selected organizer KB passages without inventing facts."""
import re

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


class QueryUnion(Section):
    tutor_extract_scope = 'full_last_chunk_union'
    include_referenced_tables = False

    def _excerpt(self, state):
        chunks = self.searcher.chunks
        indices = [i for i, chunk in enumerate(chunks) if chunk.section_id == state.section]
        if not indices:
            return ''
        full = '\n'.join(message['content'] for message in state.messages)
        selected = set()
        for query in dict.fromkeys((full, state.last_user)):
            scores = self.searcher.bm25.scores(query)
            selected.add(max(indices, key=lambda i: scores[i]))
        if self.include_referenced_tables:
            reference = re.compile(r'\(표:\s*' + re.escape(state.section) + r'-table\d+\.csv\)')
            if any(reference.search(chunks[i].text) for i in selected):
                selected.update(i for i in indices if getattr(chunks[i], 'heading', '') == '표')
        # Keep original KB order, including tables, and never duplicate a chunk.
        return '\n\n'.join(chunks[i].text for i in indices if i in selected)


class QueryUnionTables(QueryUnion):
    tutor_extract_scope = 'full_last_chunk_union_with_tables'
    include_referenced_tables = True
