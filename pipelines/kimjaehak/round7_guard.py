"""Independent guard feature and verdict-aggregation ablations on A."""
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .classifier_matrix import CharMLP50
from .classifier_models import CategoryModel, public_data
from .context_output import Pipeline as ContextOutput
from .peace_guard_fusion import isolated_state
from .span_tagger import refine_spans
from .voted_output import consensus_index


class BoundaryModel(CategoryModel):
    def fit(self, texts, labels, vectors=None):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),
            min_df=2,max_features=60000,sublinear_tf=True)
        with threadpool_limits(limits=1):
            x=self.vectorizer.fit_transform(texts)
            self.model=MLPClassifier(hidden_layer_sizes=(self.width,),activation='relu',
                solver='lbfgs',alpha=self.alpha,max_iter=600,max_fun=30000,
                random_state=self.seed,tol=1e-5,early_stopping=False)
            self.model.fit(x,labels,sample_weight=compute_sample_weight('balanced',labels))
        return self


class BoundaryGate(CharMLP50):
    classifier_representation = 'char_wb_2_5'
    # Frozen top-50% nonharmful OOF point, same grouped folds as baseline.
    classifier_threshold = .8720184128780901

    def __init__(self, searcher, llm):
        super().__init__(searcher,llm)
        rows,texts,_,_=public_data()
        self.matrix_model=BoundaryModel('char_mlp',self.classifier_seed,
            self.classifier_width,self.classifier_alpha,self.classifier_c).fit(
                texts,[r['labels']['category'] for r in rows])


def verdict_first_index(candidate, votes):
    """Vote on grounded first; select an original span verdict within the majority."""
    if not votes:
        raise ValueError('At least one completed vote required')
    bad=[i for i,v in enumerate(votes) if v]
    clean=[i for i,v in enumerate(votes) if not v]
    if len(bad)==len(clean):
        return consensus_index(candidate,votes)
    selected=bad if len(bad)>len(clean) else clean
    return selected[consensus_index(candidate,[votes[i] for i in selected])]


class VerdictFirst(CharMLP50):
    guard_verdict_first = True

    async def _fusion_ground(self, state):
        trials=[isolated_state(state) for _ in range(self.output_votes)]
        completed=set()
        async def one(index,trial):
            try:
                await ContextOutput._check_output(self,trial)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                trial.errors.append(f'fusion_ground_vote:{type(exc).__name__}')
            completed.add(index)
        tasks=[asyncio.create_task(one(i,t)) for i,t in enumerate(trials)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            finished=[t for i,t in enumerate(trials) if i in completed]
            for t in finished:state.errors.extend(t.errors)
            good=[t for t in finished if not t.errors]
            if good:
                votes=[t.unsupported for t in good]
                chosen=verdict_first_index(state.candidate_answer or '',votes)
                old=consensus_index(state.candidate_answer or '',votes)
                state.debug['r7_verdict_changed']=votes[chosen]!=votes[old]
                state.unsupported=good[chosen].unsupported
            else:state.unsupported=[]
            state.grounded=not state.unsupported
            state.debug['fusion_completed_ground_votes']=len(good)
            if self.guard_restore_span_refinement and state.unsupported:
                try:
                    predicted,confidence=self.span_tagger.predict(state.candidate_answer or '')
                    spans,changed=refine_spans(state.candidate_answer or '',state.unsupported,
                        predicted,confidence,self.learned_span_threshold)
                    if changed:
                        state.unsupported=spans
                        state.debug['learned_span']=True
                except Exception as exc:
                    state.errors.append(f'fusion_span:{type(exc).__name__}')
