"""Batch retrieval and classifier embeddings within one guard request only."""
from copy import copy
from .calibrated import low_signal
from .classifier_models import CategoryModel
from .embedding_identity import same_embedding_model
from .round7_hybrid import ConcatGate, BlendGate


class BatchProxy:
    def __init__(self, base, state):self.base,self.state=base,state

    def __getattr__(self, name):return getattr(self.base,name)

    async def embed(self, texts, *args, **kwargs):
        if len(texts)!=1:
            return await self.base.embed(texts,*args,**kwargs)
        last=self.state.last_user
        requested=list(texts) if texts[0]==last else [texts[0],last]
        vectors=await self.base.embed(requested,*args,**kwargs)
        # Validate before making an independently named classifier vector
        # available. A bad batch cannot silently classify from retrieval text.
        CategoryModel._vectors(vectors,len(requested))
        self.state._r7_classifier_vector=list(vectors[-1])
        self.state._r7_embedding_batched=True
        return [vectors[0]]


class ClassifierProxy:
    def __init__(self, base, state):self.base,self.state=base,state

    def __getattr__(self, name):return getattr(self.base,name)

    async def embed(self, texts, *args, **kwargs):
        vector=getattr(self.state,'_r7_classifier_vector',None)
        if list(texts)==[self.state.last_user] and vector is not None:
            self.state._r7_embedding_reused=True
            return [vector]
        return await self.base.embed(texts,*args,**kwargs)


class EmbeddingBatch:
    classifier_embedding_batched=True

    async def _search(self, state, extra=''):
        if (getattr(state,'_material_budget_guard',False) is not True
                or low_signal(state.last_user) or not self.searcher.has_embeddings
                or not self.llm.can_embed or not same_embedding_model(
                    self.llm.embedding_model,self.classifier_embedding_model)):
            return await super()._search(state,extra)
        current=copy(self);current.llm=BatchProxy(self.llm,state)
        return await super(EmbeddingBatch,current)._search(state,extra)

    async def _classify(self, state):
        if (getattr(state,'_material_budget_guard',False) is not True
                or getattr(state,'_r7_classifier_vector',None) is None):
            return await super()._classify(state)
        current=copy(self);current.llm=ClassifierProxy(self.llm,state)
        try:
            return await super(EmbeddingBatch,current)._classify(state)
        finally:
            state.debug['r7_embedding_reused']=getattr(state,'_r7_embedding_reused',False)
            # Ephemeral vectors never enter logs, exported responses or storage.
            del state._r7_classifier_vector


class BatchedConcat(EmbeddingBatch,ConcatGate):
    pass


class BatchedBlend(EmbeddingBatch,BlendGate):
    pass
