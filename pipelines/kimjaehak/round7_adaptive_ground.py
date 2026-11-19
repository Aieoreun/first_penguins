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

    async def _fusion_ground(self,state):
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
            await asyncio.gather(*launch(range(2)))
            candidate=state.candidate_answer or ''
            if (not trials[0].errors and not trials[1].errors
                    and covered_positions(candidate,trials[0].unsupported)
                    ==covered_positions(candidate,trials[1].unsupported)):
                state.debug['r7_ground_early']=True
            else:
                await asyncio.gather(*launch(range(2,self.output_votes)))
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
