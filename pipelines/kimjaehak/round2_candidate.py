"""Combine dense public policy examples with uncertain tutor source selection."""
from .round2 import Sources
from .input_examples import Dense


class Pipeline(Sources, Dense):
    """Keep ContextFast's output guard; change input examples and tutor sources."""
    pass
