"""Deterministic draft selection without labels, model calls, or stored answers."""
from collections import Counter
import math
import re
import unicodedata


def select(answers, method):
    if not answers:
        raise ValueError('No candidate answers')
    texts=[re.sub(r'\s+','',unicodedata.normalize('NFKC',text).casefold()) for text in answers]
    if method=='longest':
        return max(range(len(texts)),key=lambda i:len(texts[i]))
    if method!='consensus':
        raise ValueError('Unknown selection method')
    features=[Counter(text[i:i+n] for n in (2,3) for i in range(max(0,len(text)-n+1))) for text in texts]
    norms=[math.sqrt(sum(v*v for v in f.values())) or 1 for f in features]
    def similarity(i,j):
        if texts[i]==texts[j]:return 1.
        return sum(v*features[j].get(k,0) for k,v in features[i].items())/(norms[i]*norms[j])
    return max(range(len(texts)),key=lambda i:sum(similarity(i,j) for j in range(len(texts)) if i!=j))
