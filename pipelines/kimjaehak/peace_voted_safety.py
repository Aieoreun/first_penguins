"""Three independent safety votes within the inherited single stage budget."""
import asyncio

from .peace_guard_fusion import Pipeline as Fusion, isolated_state
from .peace_conditioned_safety import ConditionedSafety


class SafetyVote:
    guard_safety_votes = 3

    async def _fusion_safety(self, state):
        state.safe = False
        trials = [isolated_state(state) for _ in range(self.guard_safety_votes)]
        completed = set()

        async def one(index, trial):
            try:
                await super(SafetyVote, self)._fusion_safety(trial)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                trial.errors.append(f'fusion_safety_vote:{type(exc).__name__}')
            completed.add(index)

        tasks = [asyncio.create_task(one(index, trial)) for index, trial in enumerate(trials)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            finished = [trial for index, trial in enumerate(trials) if index in completed]
            for trial in finished:
                state.errors.extend(trial.errors)
            valid = [trial.safe for trial in finished
                     if not trial.errors and isinstance(trial.safe, bool)]
            # False on ties or no valid results; a failed call is not a vote.
            state.safe = sum(valid) > len(valid) / 2
            state.debug['fusion_completed_safety_votes'] = len(valid)


class VotedSafety(SafetyVote, Fusion):
    pass


class VotedConditioned(SafetyVote, ConditionedSafety):
    pass
