"""Stage-local reasoning ablations; existing A/C classes remain untouched.

Shallow request-local copies share the original client and request semaphore.
The client adapter changes reasoning effort and, only for explicit budget
rescues, the token cap. Prompts, timeouts, embeddings and cancellation stay intact.
"""
from copy import copy

from .classifier_matrix import CharMLP50
from .neural_peace_guard import CheckBudgets, _guard_deadline


class EffortClient:
    def __init__(self, client, effort, token_floor=None):
        self.client = client
        self.effort = effort
        self.token_floor = token_floor

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def chat(self, *args, **kwargs):
        kwargs = {**kwargs, 'effort': self.effort}
        if self.token_floor is not None:
            kwargs['max_tokens'] = max(kwargs.get('max_tokens', 0), self.token_floor)
        return await self.client.chat(*args, **kwargs)


class StageEffort:
    reasoning_guard_input = None
    reasoning_guard_ground = None
    reasoning_guard_safety = None
    reasoning_tutor_input = None
    reasoning_tutor_source = None
    reasoning_guard_input_tokens = None
    reasoning_guard_ground_tokens = None

    def _reasoning_copy(self, effort, token_floor=None):
        result = copy(self)
        result.llm = EffortClient(self.llm, effort, token_floor)
        return result

    async def _ask(self, state, name, system, user, effort, timeout):
        guard = getattr(state, '_material_budget_guard', None)
        target = self.reasoning_guard_input if guard is True else (
            self.reasoning_tutor_input if guard is False else None)
        if guard is True and self.reasoning_guard_input_tokens is not None:
            selected = self._reasoning_copy(target or effort, self.reasoning_guard_input_tokens)
            return await super(StageEffort, selected)._ask(state, name, system, user, target or effort, timeout)
        return await super()._ask(state, name, system, user, target or effort, timeout)

    async def _ground_response(self, payload):
        # C's guard context starts before queue/search. Never change tutor calls.
        if self.reasoning_guard_ground and _guard_deadline.get() is not None:
            selected = self._reasoning_copy(self.reasoning_guard_ground, self.reasoning_guard_ground_tokens)
            return await super(StageEffort, selected)._ground_response(payload)
        return await super()._ground_response(payload)

    async def _fusion_safety(self, state):
        if self.reasoning_guard_safety:
            selected = self._reasoning_copy(self.reasoning_guard_safety)
            return await super(StageEffort, selected)._fusion_safety(state)
        return await super()._fusion_safety(state)

    async def _source_response(self, state, ids, timeout):
        if self.reasoning_tutor_source and getattr(state, '_material_budget_guard', None) is False:
            selected = self._reasoning_copy(self.reasoning_tutor_source)
            return await super(StageEffort, selected)._source_response(state, ids, timeout)
        return await super()._source_response(state, ids, timeout)


class GuardInputMedium(StageEffort, CheckBudgets):
    reasoning_guard_input = 'medium'


class GuardGroundMedium(StageEffort, CheckBudgets):
    reasoning_guard_ground = 'medium'


class GuardSafetyMedium(StageEffort, CheckBudgets):
    reasoning_guard_safety = 'medium'


class GuardInputMediumBudget(GuardInputMedium):
    reasoning_guard_input_tokens = 2000


class GuardGroundMediumBudget(GuardGroundMedium):
    reasoning_guard_ground_tokens = 2000


class TutorInputMedium(StageEffort, CharMLP50):
    reasoning_tutor_input = 'medium'


class TutorSourceMedium(StageEffort, CharMLP50):
    reasoning_tutor_source = 'medium'


class TutorAnswerMedium(CharMLP50):
    # Used by the inherited v1 text-generation path. Source selection, exact KB
    # excerpts and the separate contextual-refusal call remain unchanged.
    answer_effort = 'medium'
