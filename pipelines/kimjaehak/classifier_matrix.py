"""Guard-only scratch category replacement with a common LLM fallback.

The original tutor, retrieval, grounding, safety and harmful-subtype model are
preserved. Never fall through to the old category gates after a new gate abstains:
that would hide the effect of the replacement behind a second trained model.
"""
import asyncio

import numpy as np

from core.schema import Label, parse_label
from .calibrated import low_signal
from .classifier_models import CategoryModel, public_data
from .peace_cancel_ground import FastCancelBlock
from .voted import Pipeline as Voted


class MatrixGate(FastCancelBlock):
    classifier_kind = 'char_lr'
    classifier_seed = 20261118
    classifier_width = 32
    classifier_alpha = .1
    classifier_c = 3000
    classifier_threshold = .95
    classifier_embedding_timeout = 3.0
    classifier_text_mode = 'last_user'
    classifier_common_llm_fallback = True

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        rows, texts, vectors, self.classifier_embedding_model = public_data()
        self.matrix_model = CategoryModel(self.classifier_kind, self.classifier_seed,
            self.classifier_width, self.classifier_alpha, self.classifier_c).fit(
                texts, [r['labels']['category'] for r in rows], vectors)

    async def _classify(self, state):
        if getattr(state, '_material_budget_guard', False) is not True:
            return await super()._classify(state)
        if low_signal(state.last_user):
            state.label = Label.LOW_SIGNAL_INPUT
            state.section = state.chunks[0].section_id if state.chunks else None
            return
        vectors = None
        if self.classifier_kind.startswith('dense'):
            if (not self.llm.can_embed or self.llm.embedding_model != self.classifier_embedding_model):
                state.errors.append('matrix_embedding:UnavailableOrModelMismatch')
                return await Voted._classify(self, state)
            try:
                # This is a classification vector, not the retrieval query that
                # may also contain conversation history and a candidate answer.
                vectors = await asyncio.wait_for(self.llm.embed([state.last_user],
                    timeout=self.classifier_embedding_timeout), self.classifier_embedding_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.errors.append(f'matrix_embedding:{type(exc).__name__}')
                return await Voted._classify(self, state)
        try:
            p = self.matrix_model.probabilities([state.last_user], vectors)[0]
            if (p.shape != (len(self.matrix_model.classes),) or not np.isfinite(p).all()
                    or np.any(p < 0) or not np.isclose(p.sum(), 1)):
                raise ValueError('Invalid category probabilities')
            index = int(p.argmax()); confidence = float(p[index])
            category = str(self.matrix_model.classes[index])
            label = parse_label(category)
            if confidence >= self.classifier_threshold:
                if category == 'harmful_content':
                    sub = self.harmful_classifier.predict_proba(
                        self.harmful_vectorizer.transform([state.last_user]))[0]
                    sub_index = int(sub.argmax())
                    action = str(self.harmful_classifier.classes_[sub_index])
                    label = (Label.HARMFUL_REQUEST if action == 'refuse' else Label.HARMFUL_SPEECH
                             if action == 'redirect' else None) if sub[sub_index] >= self.learned_harmful_threshold else None
                if label is not None:
                    state.label = label
                    state.section = state.chunks[0].section_id if state.chunks else None
                    state.debug.update(matrix_gate=True, matrix_confidence=confidence)
                    return
        except Exception as exc:
            state.errors.append(f'matrix_classifier:{type(exc).__name__}')
        state.debug['matrix_fallback'] = True
        await Voted._classify(self, state)


class CharLR(MatrixGate):
    classifier_kind = 'char_lr'
    # Fixed cm02 OOF top-25% nonharmful coverage; not a calibrated probability.
    classifier_threshold = 0.991844085969787


class DenseLR(MatrixGate):
    classifier_kind = 'dense_lr'
    # Fixed cm02 OOF top-25% nonharmful coverage; not a calibrated probability.
    classifier_threshold = 0.9962452194765969


class CharMLP(MatrixGate):
    classifier_kind = 'char_mlp'
    # Fixed cm02 OOF top-25% nonharmful coverage; not a calibrated probability.
    classifier_threshold = 0.9641535987031538


class DenseMLP(MatrixGate):
    classifier_kind = 'dense_mlp'
    # Fixed cm02 OOF top-25% nonharmful coverage; not a calibrated probability.
    classifier_threshold = 0.9725386873537929
