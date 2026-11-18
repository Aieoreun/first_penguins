"""Stop in-flight grounding only once reliable input classification blocks."""
from .peace_gate_threshold import FastGate90
from .peace_guard_variants import RequestedAction


class FastCancelBlock(FastGate90):
    guard_cancel_ground_after_block = True


class FastCancelRequested(FastCancelBlock, RequestedAction):
    """Same scheduling/gates; Peace requested-action rubric for guard fallback."""
    pass
