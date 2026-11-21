"""실험용. v11(jh_parallel_guard)에서 인용하는 분량만 줄인다.

v11은 고른 섹션을 통째로 인용하되 소단원을 관련도(BM25) 순으로 줄 세운다.
여기서는 그 순서의 위에서 SHORT_TOP개(기본 1)만 인용한다. 나머지 흐름은 v11 그대로다.
"""
import os

from pipelines import load

TOP = int(os.environ.get("SHORT_TOP", "1"))


class Pipeline(load("jh_parallel_guard")):
    def _excerpt(self, state):
        chunks = self.searcher.chunks
        indices = [i for i, chunk in enumerate(chunks) if chunk.section_id == state.section]
        if not indices:
            return ''
        full = '\n'.join(message['content'] for message in state.messages)
        relevance = {i: 0.0 for i in indices}
        for query in dict.fromkeys((full, state.last_user)):
            scores = self.searcher.bm25.scores(query)
            scale = max(scores[i] for i in indices) or 1.0
            for i in indices:
                relevance[i] += scores[i] / scale
        ordered = sorted(indices, key=lambda i: -relevance[i])[:TOP]
        title = chunks[indices[0]].text.split('\n', 1)[0]
        parts = [chunks[i].text.split('\n', 1)[1] if '\n' in chunks[i].text else ''
                 for i in ordered]
        return title + '\n' + '\n\n'.join(part for part in parts if part)
