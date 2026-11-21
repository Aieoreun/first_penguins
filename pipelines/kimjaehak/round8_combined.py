"""Factorial candidates; existing B and every individual ablation are retained."""
from .round8_span_veto import SpanVeto
from .round8_output_veto import OutputVeto
from .round8_quantity import QuantityValuePrompt


class LocalThree(SpanVeto,OutputVeto):
    output_votes=3


class LocalThreeQuantity(LocalThree,QuantityValuePrompt):
    pass
