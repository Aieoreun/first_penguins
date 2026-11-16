"""Tutor source selection with a bounded last-question dense retrieval view."""
import asyncio
import time

from pipelines.v1.pipeline import QUERY_MAX_CHARS
from .focused_excerpt import Selected as SelectedFocused


class Pipeline(SelectedFocused):
    source_last_dense = True
    source_last_dense_timeout = 3.0
    source_last_dense_sections = 2

    async def _candidate_sections(self, state):
        base = await super()._candidate_sections(state)
        remaining = getattr(state, '_source_extract_deadline', 0) - time.monotonic()
        if remaining < 4 or not self.searcher.has_embeddings or not self.llm.can_embed:
            return base
        timeout = min(3.0, self.source_last_dense_timeout, remaining - 1)
        try:
            vectors = await asyncio.wait_for(
                self.llm.embed([state.last_user[:QUERY_MAX_CHARS]], timeout), timeout)
            if not vectors or not vectors[0]:
                return base
            chunks = self.searcher.search(state.last_user, vectors[0], bm25_top=0, embedding_top=10)
            dense = list(dict.fromkeys(chunk.section_id for chunk in chunks))[:self.source_last_dense_sections]
            return list(dict.fromkeys(base[:2] + dense + base[2:]))[:self.source_extract_sections]
        except Exception as exc:
            # Keep the original pool on any embedding/search failure. Cancellation
            # is deliberately not caught, and no request text is retained here.
            state.errors.append(f'source_last_dense:{type(exc).__name__}')
            return base
