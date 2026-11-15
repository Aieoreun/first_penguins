"""Allocate five of the seven classification calls to the block decision."""
from .voted import Pipeline as Voted


class Pipeline(Voted):
    short_votes = 5
    other_vote_count = 1
