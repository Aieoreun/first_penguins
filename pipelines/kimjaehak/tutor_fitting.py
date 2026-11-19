"""Stronger public category fitting for tutor, with an unchanged guard instance."""
from .guard_fitted import ContextFast
from .tutor_split import Pipeline as SeparateEndpoints


class StrongerTutor(ContextFast):
    learned_gate_c = 3000


class Pipeline(SeparateEndpoints):
    tutor_type = StrongerTutor
    endpoint_variants = {'tutor':'contextfast_tutor_c3000', 'guard':'jh_guard_fitted_contextfast'}
