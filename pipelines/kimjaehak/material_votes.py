"""Three parallel material checks only for the tutor fallback classifier."""
import asyncio

from .guard_fitted import ContextFast


def select_material_vote(answers):
    """Vote on missing support; keep the first winning response unchanged."""
    valid = [answer for answer in answers
             if isinstance(answer, dict) and isinstance(answer.get('quote'), str)]
    if not valid:
        return {}
    unsupported = [not answer['quote'].strip() for answer in valid]
    missing = sum(unsupported)
    winner = unsupported[0] if missing * 2 == len(valid) else missing * 2 > len(valid)
    return next(answer for answer, direction in zip(valid, unsupported) if direction == winner)


class Pipeline(ContextFast):
    tutor_material_votes = 3

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        # Classification can replace state.debug, so endpoint identity lives
        # independently on this request's state, never on the shared pipeline.
        state._material_votes_guard = guard
        return state

    async def _ask(self, state, name, system, user, effort, timeout):
        original = super()._ask
        if name != 'material' or getattr(state, '_material_votes_guard', False):
            return await original(state, name, system, user, effort, timeout)
        tasks = [asyncio.create_task(original(state, name, system, user, effort, timeout))
                 for _ in range(self.tutor_material_votes)]
        try:
            # gather preserves call order even when completion order differs.
            # The parent handles normal API/parse errors on the original state.
            answers = await asyncio.gather(*tasks)
        except BaseException:
            # Also stop siblings if one child is cancelled independently.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return select_material_vote(answers)
