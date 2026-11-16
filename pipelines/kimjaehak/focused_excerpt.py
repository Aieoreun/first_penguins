"""Keep the entire selected section, but show relevant subsections first."""
from .extractive_answer import Section


class Pipeline(Section):
    tutor_extract_scope = 'relevance_ordered_section'

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
        ordered = sorted(indices, key=lambda i: -relevance[i])
        # Match Searcher.section_text's title/body construction, changing only
        # subsection order. Each code block and CSV table remains intact.
        title = chunks[indices[0]].text.split('\n', 1)[0]
        parts = [chunks[i].text.split('\n', 1)[1] if '\n' in chunks[i].text else ''
                 for i in ordered]
        return title + '\n' + '\n\n'.join(part for part in parts if part)
