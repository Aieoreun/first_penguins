"""Compose tutor material budgets with source-only streaming selection."""
from .material_budget import Pipeline as MaterialBudget
from .source_stream import Pipeline as SourceStream


class Pipeline(MaterialBudget, SourceStream):
    pass
