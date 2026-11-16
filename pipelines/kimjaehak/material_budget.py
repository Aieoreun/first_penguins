"""Tutor-only material/prepare budgets; unchanged 26s request deadline."""
from copy import copy
import time

from pipelines.kimjaehak.focused_excerpt import Selected as SelectedFocused


class Pipeline(SelectedFocused):
    tutor_material_timeout = 13.0
    tutor_prepare_limit = 16.0

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._material_budget_guard = guard
        return state

    async def _prepare_within(self, state, extra=''):
        if getattr(state, '_material_budget_guard', True):
            return await super()._prepare_within(state, extra)
        request_pipeline = copy(self)
        request_pipeline.prepare_limit = self.tutor_prepare_limit
        return await super(Pipeline, request_pipeline)._prepare_within(state, extra)

    async def _ask(self, state, name, system, user, effort, timeout):
        deadline = getattr(state, '_source_extract_deadline', None)
        if (name == 'material' and not getattr(state, '_material_budget_guard', True)
                and deadline is not None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Match the parent's unsuccessful-call dict/error contract.
                # No call/retry is started after the request budget is gone.
                state.errors.append('material:TimeoutError')
                return {}
            timeout = min(self.tutor_material_timeout, remaining)
        return await super()._ask(state, name, system, user, effort, timeout)


TutorMaterialBudget = Pipeline
