"""Use an unambiguous final tutor question for retrieval, preserving dialogue."""
from dataclasses import replace

from core.search import tokens
from .guard_fitted import ContextFast


def section_scores(searcher, query):
    scores = {}
    for chunk, score in zip(searcher.chunks, searcher.bm25.scores(query)):
        scores[chunk.section_id] = max(scores.get(chunk.section_id, 0), score)
    return sorted(scores.items(), key=lambda item: -item[1])


def query_features(searcher, messages):
    last = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
    full = section_scores(searcher, '\n'.join(m['content'] for m in messages))
    ranked = section_scores(searcher, last)
    first = ranked[0][1] if ranked else 0
    second = ranked[1][1] if len(ranked) > 1 else 0
    return {'base': full[0][0] if full else None,
            'last': ranked[0][0] if ranked else None, 'score': first,
            'gap': (first - second) / first if first else 0,
            'terms': len(set(tokens(last)))}


def use_last(features, min_score=12, min_gap=.3, min_terms=4):
    return (features['base'] != features['last'] and features['score'] >= min_score
            and features['gap'] >= min_gap and features['terms'] >= min_terms)


class Pipeline(ContextFast):
    tutor_query_gate = True

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._query_guard_endpoint = guard
        return state

    async def _search(self, state, extra=''):
        if getattr(state, '_query_guard_endpoint', False) or not state.history:
            return await super()._search(state, extra)
        features = query_features(self.searcher, state.messages)
        if not use_last(features):
            return await super()._search(state, extra)
        selected = replace(state, messages=[{'role': 'user', 'content': state.last_user}],
                           query='', chunks=[], errors=[], debug=dict(state.debug))
        await super()._search(selected, extra)
        state.chunks = selected.chunks
        state.errors.extend(selected.errors)
        state.debug['last_query_gate'] = True
