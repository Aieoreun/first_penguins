"""Skip last-question dense lookup only for fixed strong lexical evidence."""
from pathlib import Path
import re

import core.kb
from pipelines.kimjaehak.last_dense_source import Pipeline as LastDense


KB_DIR = Path(core.kb.__file__).resolve().parents[1] / 'kb'
_TERM = re.compile(r'[가-힣]{4,}|(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]{2,}(?![A-Za-z0-9_])')


def _matcher(term):
    pattern = re.escape(term)
    if not re.fullmatch(r'[가-힣]+', term):
        pattern = r'(?<![A-Za-z0-9_])' + pattern + r'(?![A-Za-z0-9_])'
    return re.compile(pattern, re.I)


def build_anchors(kb_dir, searcher):
    """KB-only terms, with rarity measured across full runtime section text."""
    terms = set()
    for path in sorted(Path(kb_dir).glob('*/*.md')):
        fence = None
        for line in path.read_text(encoding='utf-8').splitlines():
            marker = re.match(r'^\s*(`{3,}|~{3,})', line)
            if marker:
                char = marker.group(1)[0]
                if fence is None:
                    fence = char
                elif fence == char:
                    fence = None
                continue
            if fence is not None:
                continue
            fragments = []
            heading = re.match(r'^#{1,6}\s+(.+)$', line)
            if heading:
                fragments.append(heading.group(1))
            fragments.extend(match.group(2) for match in re.finditer(r'(`+)([^`\n]+)\1', line))
            for fragment in fragments:
                terms.update(match.group().lower() for match in _TERM.finditer(fragment))
    sections = {sid: searcher.section_text(sid) for sid in searcher.section_ids()}
    anchors = {}
    for term in sorted(terms):
        pattern = _matcher(term)
        present = frozenset(sid for sid, text in sections.items() if pattern.search(text))
        if 1 <= len(present) <= 2:
            anchors[term] = (pattern, present)
    return anchors


class Pipeline(LastDense):
    source_conditional_dense = True

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self._lexical_anchors = build_anchors(KB_DIR, searcher)

    def _lexical_strong(self, query):
        best = {}
        for chunk, score in zip(self.searcher.chunks, self.searcher.bm25.scores(query)):
            best[chunk.section_id] = max(best.get(chunk.section_id, 0.0), score)
        ranked = sorted(best, key=lambda sid: -best[sid])
        if not ranked or best[ranked[0]] <= 0:
            return False
        top = ranked[0]
        runnerup = best[ranked[1]] if len(ranked) > 1 else 0.0
        if best[top] < 1.5 * runnerup:
            return False
        return any(top in sections and pattern.search(query)
                   for pattern, sections in self._lexical_anchors.values())

    async def _candidate_sections(self, state):
        if self._lexical_strong(state.last_user):
            return self._source_candidates(state)
        return await super()._candidate_sections(state)
