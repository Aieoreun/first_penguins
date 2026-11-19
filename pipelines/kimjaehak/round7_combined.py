"""Combine independently screened guard changes; retain A's tutor path."""
from .round7_adaptive_ground import AdaptiveGround
from .round7_batch import BatchedBlend


class BlendedAdaptive(BatchedBlend,AdaptiveGround):
    # Dense classifier attributes must precede CharMLP50 in this MRO. The batch
    # proxies still use request-local state and adaptive votes affect only guard.
    pass
