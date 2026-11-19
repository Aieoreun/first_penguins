"""Isolate the number of KB sections in the current five-vote output guard."""
from .guard_fitted import ContextFast


class Two(ContextFast):
    output_sections = 2


class Three(ContextFast):
    output_sections = 3
