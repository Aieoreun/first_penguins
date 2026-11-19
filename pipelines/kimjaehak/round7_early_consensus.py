"""Parallel five-vote grounding with an early unanimous completed-three exit."""
import asyncio
from .round7_adaptive_ground import AdaptiveGround
from .voted_output import covered_positions


class EarlyConsensus(AdaptiveGround):
    guard_adaptive_initial_votes=3
    guard_vote_schedule='parallel_five_early_three'

    async def _schedule_ground_votes(self,state,trials,launch):
        tasks=launch(range(self.output_votes));pending=set(tasks)
        candidate=state.candidate_answer or ''
        while pending:
            _,pending=await asyncio.wait(pending,return_when=asyncio.FIRST_COMPLETED)
            finished=[trial for task,trial in zip(tasks,trials) if task.done() and not task.cancelled()]
            if len(finished)<3 or any(trial.errors for trial in finished):continue
            masks=[covered_positions(candidate,trial.unsupported) for trial in finished]
            if all(mask==masks[0] for mask in masks[1:]):
                state.debug['r7_ground_early']=bool(pending)
                # Parent finally cancels and joins every still-running child,
                # retaining all completed votes in original creation order.
                return
