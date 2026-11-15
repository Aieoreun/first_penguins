"""Shorten only the material-classification context, retaining all search hits."""
from .adaptive import Pipeline as Adaptive
from .voted import Pipeline as Voted


class Pipeline(Voted):
    material_max_chunks = 5


class AdaptivePipeline(Adaptive, Pipeline):
    pass
