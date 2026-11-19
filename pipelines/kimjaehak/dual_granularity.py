"""Tutor source selection: BM25 + existing long + new short embedding hits."""
from copy import copy

from core.search import load_vectors
from .dual_index import DenseView, candidate_pool, index_path, short_chunks
from .material_budget import Pipeline as MaterialBudget


class _CaptureSearch:
    """Request-local adapter: preserve the parent's query/embedding/fallback path."""
    def __init__(self, base):
        self.base = base
        self.query, self.vector = '', None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def search(self, query, query_vector=None, **kwargs):
        self.query, self.vector = query, query_vector
        return self.base.search(query, query_vector, **kwargs)


class Six(MaterialBudget):
    source_dual_granularity = True
    source_extract_sections = 6

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self.long_view = DenseView(searcher.chunks, searcher.vectors)
        chunks = short_chunks(searcher.chunks)
        self.short_view = DenseView(chunks, load_vectors(index_path(), llm.embedding_model, chunks),
                                    dimension=self.long_view.matrix.shape[1])

    async def _search(self, state, extra=''):
        if getattr(state, '_material_budget_guard', True):
            return await super()._search(state, extra)
        request_pipeline = copy(self)
        capture = _CaptureSearch(self.searcher)
        request_pipeline.searcher = capture
        await super(Six, request_pipeline)._search(state, extra)
        # Input checks receive exactly the original BM25+long chunks. The new
        # short view only expands the subsequent tutor source-selection pool.
        bm25 = self.searcher.search(capture.query, bm25_top=10, embedding_top=0)
        long, short = [], []
        if capture.vector is not None:
            try:
                long = self.long_view.search(capture.vector)
                short = self.short_view.search(capture.vector)
            except ValueError:
                state.errors.append('dual_embed:ValueError')
        state._dual_routes = (bm25, long, short)

    def _source_candidates(self, state):
        last = self.searcher.search(state.last_user, bm25_top=1, embedding_top=0)
        pool = candidate_pool(state.section, last, getattr(state, '_dual_routes', (state.chunks,)),
                              state.chunks)
        return pool[:self.source_extract_sections]


class All(Six):
    source_extract_sections = None  # all retrieved sections, not the entire KB
