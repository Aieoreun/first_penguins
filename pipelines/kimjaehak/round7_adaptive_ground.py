"""Reduce redundant guard grounding calls; retain completed-vote semantics."""
import asyncio

from .classifier_matrix import CharMLP50
from .context_output import Pipeline as ContextOutput
from .peace_guard_fusion import isolated_state
from .span_tagger import refine_spans
from .voted_output import consensus_index, covered_positions


class GroundThree(CharMLP50):
    output_votes=3


class AdaptiveGround(CharMLP50):
    guard_adaptive_ground=True
    output_votes=5
    guard_adaptive_initial_votes=2

    async def _fusion_ground(self,state):
        initial=self.guard_adaptive_initial_votes
        if initial not in (2,3):raise ValueError("Unsupported initial vote count")
        trials=[isolated_state(state) for _ in range(self.output_votes)]
        completed=set()
        tasks=[]

        async def one(index):
            trial=trials[index]
            try:
                await ContextOutput._check_output(self,trial)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                trial.errors.append(f'fusion_ground_vote:{type(exc).__name__}')
            completed.add(index)

        def launch(indices):
            batch=[asyncio.create_task(one(index)) for index in indices]
            tasks.extend(batch)
            return batch

        try:
            await self._schedule_ground_votes(state,trials,launch)
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            # Same creation-order tie break and error exclusion as A. Parent
            # BLOCK/deadline cancellation may arrive during either batch.
            finished=[trial for i,trial in enumerate(trials) if i in completed]
            for trial in finished:state.errors.extend(trial.errors)
            good=[trial for trial in finished if not trial.errors]
            state.unsupported=(good[consensus_index(state.candidate_answer or '',
                [trial.unsupported for trial in good])].unsupported if good else [])
            state.grounded=not state.unsupported
            state.debug['fusion_completed_ground_votes']=len(good)
            if self.guard_restore_span_refinement and state.unsupported:
                try:
                    predicted,confidence=self.span_tagger.predict(state.candidate_answer or '')
                    spans,changed=refine_spans(state.candidate_answer or '',state.unsupported,
                        predicted,confidence,self.learned_span_threshold)
                    if changed:
                        state.unsupported=spans
                        state.debug['learned_span']=True
                except Exception as exc:
                    state.errors.append(f'fusion_span:{type(exc).__name__}')

    async def _schedule_ground_votes(self,state,trials,launch):
        initial=self.guard_adaptive_initial_votes
        await asyncio.gather(*launch(range(initial)))
        candidate=state.candidate_answer or ''
        masks=[covered_positions(candidate,trial.unsupported) for trial in trials[:initial]]
        if (all(not trial.errors for trial in trials[:initial])
                and all(mask==masks[0] for mask in masks[1:])):
            state.debug['r7_ground_early']=True
        else:
            await asyncio.gather(*launch(range(initial,self.output_votes)))


class AdaptiveThree(AdaptiveGround):
    guard_adaptive_initial_votes=3
