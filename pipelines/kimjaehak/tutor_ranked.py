"""Use generic learned relevance features only for uncertain tutor sources."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from core.schema import Label
from .guard_fitted import ContextFast
from .learned_gate import ROOT
from .section_ranker import SectionFeatures


class Pipeline(ContextFast):
    tutor_section_ranker = True
    tutor_ranker_c = .1
    tutor_ranker_margin = 0.

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self.section_features = SectionFeatures(searcher)
        rows = [r for r in map(json.loads, (ROOT/'public/tutor.jsonl').read_text().splitlines())
                if len(r['labels'].get('evidence') or []) == 1]
        with threadpool_limits(limits=1):
            x = np.array([self.section_features.transform(r['messages']) for r in rows])
            y = np.array([[int(sid in r['labels']['evidence']) for sid in self.section_features.ids] for r in rows])
            self.section_ranker = make_pipeline(StandardScaler(), LogisticRegression(
                C=self.tutor_ranker_c, class_weight='balanced', max_iter=1000))
            self.section_ranker.fit(x.reshape(-1,x.shape[-1]), y.ravel())

    async def _generate(self, state):
        if (state.label in {Label.NORMAL, Label.PII_DISCLOSURE, Label.ANSWER_SEEKING}
                and not state.debug.get('learned_section')):
            try:
                x = self.section_features.transform(state.messages)
                probabilities = self.section_ranker.predict_proba(x)[:,1]
                indices = np.argsort(-probabilities, kind='stable')
                margin = float(probabilities[indices[0]] - probabilities[indices[1]]) if len(indices)>1 else 1.
                if margin >= self.tutor_ranker_margin:
                    selected = self.section_features.ids[indices[0]]
                    state.debug['tutor_ranker_changed'] = selected != state.section
                    state.section = selected
                    state.debug['tutor_ranker_margin'] = margin
            except Exception as exc:
                state.errors.append(f'tutor_ranker:{type(exc).__name__}')
        await super()._generate(state)
