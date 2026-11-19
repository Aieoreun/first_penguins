"""Guard-only scratch character/dense fusion; preserve A's tutor and output."""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .classifier_matrix import DenseMLP50, CharMLP50
from .classifier_models import CategoryModel, public_data
from .calibrated import low_signal
from core.schema import parse_label
from .voted import Pipeline as Voted


class CommonFallback(CharMLP50):
    """A's exact fallback without trained input gates, for controlled replay."""
    guard_learned_input_enabled=False

    async def _classify(self, state):
        if getattr(state,'_material_budget_guard',False) is True and not low_signal(state.last_user):
            await Voted._classify(self,state)
            state.debug['guard_unlearned_input']=True
            return
        await super()._classify(state)


class ConcatModel(CategoryModel):
    """Unit-norm char TF-IDF and unit-norm organizer vectors, one scratch MLP."""
    def fit(self, texts, labels, vectors=None):
        self.vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,5),
            min_df=2,max_features=60000,sublinear_tf=True)
        dense=self._vectors(vectors,len(texts))
        with threadpool_limits(limits=1):
            x=hstack([self.vectorizer.fit_transform(texts),csr_matrix(dense)],format='csr')
            self.model=MLPClassifier(hidden_layer_sizes=(self.width,),solver='lbfgs',alpha=self.alpha,
                max_iter=600,max_fun=30000,random_state=self.seed,tol=1e-5)
            self.model.fit(x,labels,sample_weight=compute_sample_weight('balanced',labels))
        return self

    def probabilities(self, texts, vectors=None):
        x=hstack([self.vectorizer.transform(texts),
            csr_matrix(self._vectors(vectors,len(texts)))],format='csr')
        return self.model.predict_proba(x)


class BlendModel:
    """Fixed 75% character / 25% dense probability mixture, class-aligned."""
    dense_weight=.25

    def __init__(self, seed=20261118, width=32, alpha=.1, c=3000):
        self.parameters=(seed,width,alpha,c)

    def fit(self, texts, labels, vectors=None):
        self.char=CategoryModel('char_mlp',*self.parameters).fit(texts,labels)
        self.dense=CategoryModel('dense_mlp',*self.parameters).fit(texts,labels,vectors)
        if set(self.char.classes)!=set(self.dense.classes):
            raise ValueError('Hybrid classifier class mismatch')
        self.dense_order=[list(self.dense.classes).index(c) for c in self.char.classes]
        return self

    @property
    def classes(self):return self.char.classes

    def probabilities(self, texts, vectors=None):
        a=self.char.probabilities(texts)
        b=self.dense.probabilities(texts,vectors)[:,self.dense_order]
        return (1-self.dense_weight)*a+self.dense_weight*b


class ConcatGate(DenseMLP50):
    classifier_representation='char_tfidf_plus_unit_dense_mlp'
    classifier_threshold=.9011610502863048
    classifier_hybrid_dense_weight=1.0

    def __init__(self, searcher, llm):
        # Keep the original model-identity check, embedding timeout/fallback,
        # harmful subtype and request-local guard marker. No model hot-swapping.
        super().__init__(searcher,llm)
        rows,texts,vectors,_=public_data()
        self.matrix_model=ConcatModel('dense_mlp',self.classifier_seed,
            self.classifier_width,self.classifier_alpha,self.classifier_c).fit(
                texts,[r['labels']['category'] for r in rows],vectors)


class BlendGate(DenseMLP50):
    classifier_representation='char75_dense25_probability_blend'
    classifier_threshold=.8208260969823079
    classifier_hybrid_dense_weight=.25

    def __init__(self, searcher, llm):
        super().__init__(searcher,llm)
        rows,texts,vectors,_=public_data()
        self.matrix_model=BlendModel(self.classifier_seed,self.classifier_width,
            self.classifier_alpha,self.classifier_c).fit(texts,[r['labels']['category'] for r in rows],vectors)


class FastBlendGate(BlendGate):
    """Skip dense calls only at A's frozen OOF top-25% nonharmful confidence."""
    classifier_hybrid_fast_threshold=.9641535987031538

    async def _classify(self, state):
        if getattr(state,'_material_budget_guard',False) is True and not low_signal(state.last_user):
            try:
                model=self.matrix_model.char
                p=model.probabilities([state.last_user])[0]
                if (p.shape!=(len(model.classes),) or not np.isfinite(p).all()
                        or np.any(p<0) or not np.isclose(p.sum(),1)):
                    raise ValueError('Invalid fast classifier probabilities')
                i=int(p.argmax());category=str(model.classes[i]);label=parse_label(category)
                if category!='harmful_content' and label is not None and p[i]>=self.classifier_hybrid_fast_threshold:
                    state.label=label
                    state.section=state.chunks[0].section_id if state.chunks else None
                    state.debug.update(matrix_gate=True,matrix_confidence=float(p[i]),r7_hybrid_fast=True)
                    return
            except Exception as exc:
                state.errors.append(f'hybrid_fast:{type(exc).__name__}')
        await super()._classify(state)
