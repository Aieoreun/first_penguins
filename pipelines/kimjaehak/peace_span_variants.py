"""Boundary-confidence ablations; input, grounding verdict and safety unchanged."""
from .material_budget import Pipeline as Current
from .peace_routed_safety import RoutedSafety, RoutedRequested


class CurrentSpan80(Current):
    learned_span_threshold = .8


class RoutedSpan80(RoutedSafety):
    learned_span_threshold = .8


class RoutedRequestedSpan80(RoutedRequested):
    learned_span_threshold = .8
