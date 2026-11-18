"""Tutor-only lexical and resolved-query ablations; unchanged guard search."""
from collections import Counter
from copy import copy
import math
import re

from core.search import BM25
from .material_budget import Pipeline as MaterialBudget
from .resolved_query import Minimal as ResolvedMinimal, TopicSelect


# KB course vocabulary from bsuk1 de664f2 app/kb_index.py; no bilingual aliases
# or example/label lookup table. Only morphological analysis is used.
COURSE_TERMS = (
    '다람쥐은행','다람 문구점','다람문구점','꿈틀이','꿈틀','kkumtle','Helpy','HelpyCode',
    '첫걸음 통장','첫걸음통장','새싹적금','꿈모으기 정기예금','틴즈 체크카드','도토리 포인트',
    '도토리포인트','하람','솔숲 쉼터','거름망','바이브 코딩','fruit_basket','fruit_order',
    '용돈 기입장','바른말 필터',
)
CODE_TOKEN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*')


def content_tokens(text, kiwi):
    out = [t.form.strip().lower() for t in kiwi.tokenize(text)
           if not t.tag.startswith('S') and t.form.strip()]
    for match in CODE_TOKEN.findall(text):
        if match.lower() not in out:
            out.append(match.lower())
    content = [t for t in out if len(t) >= 2 or t.isdigit()]
    return out + [a+'_'+b for a,b in zip(content,content[1:])]


class MorphBM25(BM25):
    """Core BM25 formula with a different vocabulary, not a dense model."""
    def __init__(self, texts, tokenize):
        self.tokenize = tokenize
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.length = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.length) / len(self.length) if self.length else 1.
        df = Counter(term for tf in self.tf for term in tf)
        self.idf = {term:math.log(1+(len(texts)-count+.5)/(count+.5)) for term,count in df.items()}
        self.k1,self.b = 1.5,.75

    def scores(self, query):
        terms = dict.fromkeys(self.tokenize(query))
        out = []
        for tf,length in zip(self.tf,self.length):
            score = 0.0
            for term in terms:
                frequency = tf.get(term,0)
                if frequency:
                    score += self.idf[term]*frequency*(self.k1+1)/(
                        frequency+self.k1*(1-self.b+self.b*length/self.avg))
            out.append(score)
        return out


class KiwiBigram(MaterialBudget):
    tutor_lexical_mode = 'kiwi_content_bigram'

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        from kiwipiepy import Kiwi  # optional dependency, only for this variant
        self._kiwi = Kiwi(num_workers=1)
        for term in COURSE_TERMS:
            self._kiwi.add_user_word(term.replace(' ',''),'NNP',0)
        self._tutor_searcher = copy(searcher)
        self._tutor_searcher.bm25 = MorphBM25([c.text for c in searcher.chunks],
                                            lambda text:content_tokens(text,self._kiwi))

    async def _search(self, state, extra=''):
        if getattr(state, '_material_budget_guard', True):
            return await super()._search(state, extra)
        request_pipeline = copy(self)
        request_pipeline.searcher = self._tutor_searcher
        return await super(KiwiBigram, request_pipeline)._search(state, extra)


class ResolvedFocused(ResolvedMinimal, MaterialBudget):
    """Previously tested query resolver, now paired with focused source output."""
    pass


class TopicFocused(TopicSelect, MaterialBudget):
    """Choose full dialogue or last message without generating a new query."""
    pass
