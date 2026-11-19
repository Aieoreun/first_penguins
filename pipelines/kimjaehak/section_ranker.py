"""Section pair features; trained weights never contain a question/answer table."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.search import BM25, tokens


class SectionFeatures:
    def __init__(self, searcher):
        self.searcher = searcher
        self.ids = searcher.section_ids()
        self.texts = [searcher.section_text(sid) for sid in self.ids]
        self.titles = [text.split('\n', 1)[0] for text in self.texts]
        self.section_bm25 = BM25(self.texts)
        self.title_bm25 = BM25(self.titles)
        self.char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True)
        self.char_sections = self.char.fit_transform(self.texts)
        self.char_titles = self.char.transform(self.titles)
        self.chunk_indices = [[i for i,c in enumerate(searcher.chunks) if c.section_id==sid] for sid in self.ids]
        self.token_sets = [set(tokens(t)) for t in self.texts]
        self.number_sets = [set(re.findall(r'\d+(?:\.\d+)?', t)) for t in self.texts]

    def transform(self, messages):
        full = '\n'.join(m['content'] for m in messages)
        last = next((m['content'] for m in reversed(messages) if m['role']=='user'), '')
        features = []
        for query in (full, last):
            chunk_scores = self.searcher.bm25.scores(query)
            scores = [np.asarray([max(chunk_scores[i] for i in indices) for indices in self.chunk_indices]),
                      np.asarray(self.section_bm25.scores(query)), np.asarray(self.title_bm25.scores(query)),
                      (self.char_sections @ self.char.transform([query]).T).toarray()[:,0],
                      (self.char_titles @ self.char.transform([query]).T).toarray()[:,0]]
            for values in scores:
                maximum = max(float(values.max()), 1e-12)
                # Rank is permutation-invariant: equal scores receive equal ranks.
                ranks = np.asarray([1 + np.count_nonzero(values > v) for v in values])
                features.extend((values/maximum, 1/ranks))
            query_tokens = set(tokens(query))
            total = sum(self.section_bm25.idf.get(t, 0) for t in query_tokens) or 1
            coverage = [sum(self.section_bm25.idf.get(t, 0) for t in query_tokens & terms)/total
                        for terms in self.token_sets]
            numbers = set(re.findall(r'\d+(?:\.\d+)?', query))
            numeric = [len(numbers & terms)/max(1,len(numbers)) for terms in self.number_sets]
            features.extend((coverage,numeric))
        return np.asarray(features,dtype=float).T
