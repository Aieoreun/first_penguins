"""Isolated guard ablations on the trained-input Peace fusion."""
from .peace_guard_fusion import Pipeline as Fusion
from .peace_guard_prompt import PEACE_BLOCK_SYSTEM


class RequestedAction(Fusion):
    """Use Peace's full requested-action BLOCK rubric only for guard fallback."""
    guard_requested_action = True
    guard_block_system = PEACE_BLOCK_SYSTEM

    def _block_system(self, state):
        if getattr(state, '_material_budget_guard', False) is True:
            return self.guard_block_system
        return super()._block_system(state)
