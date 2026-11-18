"""Tutor policy-category ablation; immutable char-MLP50 guard is inherited."""
import numpy as np

from core.schema import parse_label
from .calibrated import low_signal
from .classifier_matrix import CharMLP50
from .voted import Pipeline as Voted


class Selective(CharMLP50):
    tutor_mlp_threshold = 0.9560244784358727
    tutor_mlp_single_turn = True
    tutor_mlp_fallback = 'original'
    # Material membership needs the existing KB path. Keep the original
    # harmful request/speech subtype and low-signal handling as well.
    tutor_mlp_deferred = frozenset({'normal', 'out_of_material',
                                  'harmful_content', 'low_signal_input'})

    async def _classify(self, state):
        if (getattr(state, '_material_budget_guard', False) is True
                or low_signal(state.last_user)
                or (self.tutor_mlp_single_turn and len(state.messages) != 1)):
            return await super()._classify(state)
        try:
            p = self.matrix_model.probabilities([state.last_user])[0]
            if (p.shape != (len(self.matrix_model.classes),) or not np.isfinite(p).all()
                    or np.any(p < 0) or not np.isclose(p.sum(), 1)):
                raise ValueError('Invalid tutor classifier probabilities')
            index = int(p.argmax())
            category = str(self.matrix_model.classes[index])
            label = parse_label(category)
        except Exception as exc:
            state.errors.append('tutor_mlp:' + type(exc).__name__)
            return await super()._classify(state)
        if category in self.tutor_mlp_deferred or label is None:
            return await super()._classify(state)
        if p[index] >= self.tutor_mlp_threshold:
            state.label = label
            state.section = state.chunks[0].section_id if state.chunks else None
            state.debug['tutor_mlp_gate'] = True
        elif self.tutor_mlp_fallback == 'original':
            return await super()._classify(state)
        else:
            # Do not run the old category LR after an abstaining replacement.
            await Voted._classify(self, state)
            state.debug['tutor_mlp_llm'] = True
        # Preserve both inherited source-selection stages, once and in order.
        self._apply_learned_section(state)
        self._apply_tutor_section(state)


class Replace(Selective):
    # Frozen before API experiments: tutor OOF top-50% of eligible policy
    # predictions, not a calibrated probability or 50% of all live requests.
    tutor_mlp_threshold = 0.795080953722436
    tutor_mlp_single_turn = False
    tutor_mlp_fallback = 'llm'
