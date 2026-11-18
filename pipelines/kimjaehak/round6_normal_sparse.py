"""Compose tutor normal .94 classification with conservative .95 sparse search."""
from .round6_normal_gate import Pipeline as Normal94
from .round6_trusted_sparse import Pipeline as TrustedSparse


class Pipeline(Normal94, TrustedSparse):
    """Keep threshold relaxation local to classification, after search finishes.

    Normal94's request copy does not change the shared .95 gate used by
    TrustedSparse. Requests in the new [.94, .95) normal band still embed.
    """

    pass
