"""A separate learned binary head adjusts class-group probability mass."""
import numpy as np

BLOCKED=frozenset({"harmful_content","academic_dishonesty","off_topic","jailbreak_injection"})


def reweight(probabilities,classes,block_probability,weight):
    p=np.asarray(probabilities,dtype=float)
    b=np.asarray(block_probability,dtype=float)
    if p.ndim!=2 or b.shape!=(len(p),) or not 0<=weight<=1:
        raise ValueError('Invalid reweight shape or weight')
    if not np.isfinite(p).all() or not np.isfinite(b).all() or np.any(p<0) or np.any((b<0)|(b>1)) or not np.allclose(p.sum(1),1):
        raise ValueError('Invalid probabilities')
    mask=np.array([c in BLOCKED for c in classes]);old=p[:,mask].sum(1)
    if not mask.any() or mask.all() or np.any((old<=0)|(old>=1)):
        raise ValueError('Both decision groups need positive support')
    target=(1-weight)*old+weight*b;out=p.copy()
    out[:,mask]*=(target/old)[:,None]
    out[:,~mask]*=((1-target)/(1-old))[:,None]
    return out



class BinaryAdjustedModel:
    def __init__(self,category_model,decision_model,weight=1.):
        self.category_model=category_model
        self.decision_model=decision_model
        self.weight=weight
        if set(decision_model.classes)!={"allow","block"}:raise ValueError("Expected binary model")

    @property
    def classes(self):return self.category_model.classes

    def probabilities(self,texts,vectors=None):
        p=self.category_model.probabilities(texts,vectors)
        b=self.decision_model.probabilities(texts)[:,np.flatnonzero(self.decision_model.classes=="block")[0]]
        return reweight(p,self.classes,b,self.weight)
