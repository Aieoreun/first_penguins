"""Guard scheduling ablations without changing learned decisions or rubrics."""
from .peace_guard_variants import RequestedAction
from .peace_span_variants import RoutedSpan80


class FastGate(RoutedSpan80):
    guard_fast_input_wait = .005


class ParallelInput(RoutedSpan80):
    guard_vote_first = False


class FastParallelInput(FastGate):
    guard_vote_first = False


class FastRequestedParallel(FastParallelInput, RequestedAction):
    pass
