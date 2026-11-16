"""Conservative confidence controls for the public-trained category gates."""
from .guard_fitted import ContextFast


class Threshold98(ContextFast):
    learned_gate_threshold = .98


class Threshold995(ContextFast):
    learned_gate_threshold = .995
    guard_only_c = 1000
