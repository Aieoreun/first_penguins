"""Guard-only A–E ablations; A remains the original CharMLP50 class.

No model fitting or global prompt/timeout mutation. A request-local deadline
includes semaphore waiting and retrieval and is inherited by child check tasks.
"""
import asyncio
from contextvars import ContextVar
from copy import copy
import json
import time

from pipelines.v1.pipeline import CLASSIFY_EFFORT, GUARD_EFFORT
from .adaptive import needs_third
from .classifier_matrix import CharMLP50
from .peace_guard_fusion import Pipeline as Fusion
from .peace_guard_prompt import PEACE_BLOCK_SYSTEM, SAFETY_SYSTEM
from .voted import Pipeline as Voted

_guard_deadline = ContextVar('neural_peace_guard_deadline', default=None)


class ParallelSafety(CharMLP50):
    """B: independent safety starts alongside input, before its fast grace."""
    guard_safety_after_input = False
    guard_safety_launch_early = True
    guard_independent_safety = True
    guard_routed_safety = False
    guard_safety_system = SAFETY_SYSTEM
    guard_learned_input_enabled = True

    async def _fusion_safety(self, state):
        await Fusion._fusion_safety(self, state)


class CheckBudgets(ParallelSafety):
    """C: expand real call/group limits, bounded by the original request cap."""
    guard_short_timeout = 13.0
    guard_material_timeout = 16.0
    guard_ground_timeout = 18.0
    guard_safety_timeout = 13.0

    async def guard(self, body):
        token = _guard_deadline.set(time.monotonic() + self.guard_deadline)
        try:
            return await super().guard(body)
        finally:
            _guard_deadline.reset(token)

    def _guard_seconds(self, requested):
        deadline = _guard_deadline.get()
        remaining = requested if deadline is None else min(requested, deadline - time.monotonic())
        if remaining <= 0:
            raise asyncio.TimeoutError('Guard request budget exhausted')
        return remaining

    async def _ask(self, state, name, system, user, effort, timeout):
        if getattr(state, '_material_budget_guard', False) is True:
            try:
                timeout = self._guard_seconds(self.guard_material_timeout if name == 'material' else timeout)
            except asyncio.TimeoutError:
                state.errors.append(f'{name}:NoBudget')
                return {}
        return await super()._ask(state, name, system, user, effort, timeout)

    async def _vote_group(self, state, name, system, check):
        if getattr(state, '_material_budget_guard', False) is not True:
            return await super()._vote_group(state, name, system, check)
        started = time.monotonic()
        limit = self._guard_seconds(self.guard_short_timeout)
        pair = await asyncio.gather(*(self._ask(state, name, system, check,
            CLASSIFY_EFFORT, limit) for _ in range(2)))
        if needs_third(pair, other=name == 'other', unanimous=self.block_min_votes == 3):
            left = limit - (time.monotonic() - started)
            if left > .05:
                pair.append(await self._tie_vote(state, name, system, check, left))
            else:
                state.errors.append(f'{name}:tie_budget')
        return pair

    async def _ground_response(self, payload):
        if _guard_deadline.get() is None:
            return await super()._ground_response(payload)
        return await self.llm.chat(self.unsupported_system, json.dumps(payload, ensure_ascii=False),
            effort=GUARD_EFFORT, max_tokens=1500,
            timeout=self._guard_seconds(self.guard_ground_timeout), json_mode=True)

    async def _fusion_safety(self, state):
        bounded = copy(self)
        bounded.guard_safety_timeout = self._guard_seconds(self.guard_safety_timeout)
        await Fusion._fusion_safety(bounded, state)


class PeaceBlock(CheckBudgets):
    """D: exact pinned Peace input rubric, guard only."""
    guard_requested_action = True
    guard_block_system = PEACE_BLOCK_SYSTEM

    def _block_system(self, state):
        if getattr(state, '_material_budget_guard', False) is True:
            return self.guard_block_system
        return super()._block_system(state)


class UnlearnedInput(PeaceBlock):
    """E: no learned guard input gates; tutor and output span model retained."""
    guard_learned_input_enabled = False

    async def _classify(self, state):
        if getattr(state, '_material_budget_guard', False) is True:
            await Voted._classify(self, state)
            state.debug['guard_unlearned_input'] = True
            return
        await super()._classify(state)
