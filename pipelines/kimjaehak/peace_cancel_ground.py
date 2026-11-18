"""Stop in-flight grounding only once reliable input classification blocks."""
from .peace_gate_threshold import FastGate90


class FastCancelBlock(FastGate90):
    guard_cancel_ground_after_block = True
