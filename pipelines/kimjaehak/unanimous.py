"""Require three matching valid block votes; keep the existing other checks."""
from .voted import Pipeline as Voted


class Pipeline(Voted):
    block_min_votes = 3
