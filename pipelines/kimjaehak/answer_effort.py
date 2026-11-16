"""Compare low reasoning for answer generation against the minimal control."""
from .guard_fitted import ContextFast
from .source_choice import Pipeline as SourceChoice


class Low(ContextFast):
    answer_effort = "low"


class SourceLow(SourceChoice):
    answer_effort = "low"
