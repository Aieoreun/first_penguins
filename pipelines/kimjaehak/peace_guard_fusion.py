"""Independent guard checks with our input models and span refinement.

Tutor methods remain inherited from MaterialBudget. Independent safety replaces
the old blocked-refusal review in this guard path; it is an intentional semantic
change, not merely a scheduling optimization. No new training data are used.
"""
import asyncio
from copy import copy
import json
import time

from core.schema import Decision
from pipelines.v1.pipeline import CLASSIFY_EFFORT, _json_object
from pipelines.v2 import prompts
from .context_output import Pipeline as ContextOutput
from .material_budget import Pipeline as MaterialBudget
from .peace_guard_prompt import SAFETY_SYSTEM, PEACE_REFERENCE_COMMIT
from .span_tagger import refine_spans
from .voted_output import consensus_index


def isolated_state(state):
    """Preserve dynamic endpoint/deadline flags, isolate mutable outputs."""
    result = copy(state)
    result.errors = []
    result.debug = dict(state.debug)
    result.unsupported = []
    result.citations = list(state.citations)
    return result


class Pipeline(MaterialBudget):
    guard_fusion = True
    guard_fusion_reference = PEACE_REFERENCE_COMMIT
    guard_independent_safety = True
    guard_preserve_completed_ground = True
    guard_restore_span_refinement = True
    guard_vote_first = True
    guard_safety_after_input = False
    request_slots = 64
    guard_prepare_limit = 26.0
    guard_deadline = 28.0
    guard_material_timeout = 13.0
    guard_ground_timeout = 13.0
    guard_safety_timeout = 13.0
    guard_fast_input_wait = 0.0
    guard_safety_system = SAFETY_SYSTEM

    async def _collect_checks(self, state, check, material):
        if (not getattr(state, '_material_budget_guard', False)
                or not self.guard_vote_first):
            return await super()._collect_checks(state, check, material)
        block, other = await asyncio.gather(
            self._vote_group(state, 'block', prompts.BLOCK_SYSTEM, check),
            self._vote_group(state, 'other', prompts.OTHER_SYSTEM, check))
        outcome = await self._ask(state, 'material', prompts.MATERIAL_SYSTEM,
                                  material, CLASSIFY_EFFORT, self.guard_material_timeout)
        return block, other, outcome

    async def _fusion_ground(self, state):
        trials = [isolated_state(state) for _ in range(self.output_votes)]
        completed = set()

        async def one(index, trial):
            try:
                await ContextOutput._check_output(self, trial)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                trial.errors.append(f'fusion_ground_vote:{type(exc).__name__}')
            completed.add(index)

        tasks = [asyncio.create_task(one(index, trial)) for index, trial in enumerate(trials)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Preserve creation order for consensus ties, including cancellation.
            finished = [trial for index, trial in enumerate(trials) if index in completed]
            for trial in finished:
                state.errors.extend(trial.errors)
            good = [trial for trial in finished if not trial.errors]
            state.unsupported = (good[consensus_index(state.candidate_answer or '',
                [trial.unsupported for trial in good])].unsupported if good else [])
            state.grounded = not state.unsupported
            state.debug['fusion_completed_ground_votes'] = len(good)
            if self.guard_restore_span_refinement and state.unsupported:
                try:
                    predicted, confidence = self.span_tagger.predict(state.candidate_answer or '')
                    spans, changed = refine_spans(state.candidate_answer or '', state.unsupported,
                        predicted, confidence, self.learned_span_threshold)
                    if changed:
                        state.unsupported = spans
                        state.debug['learned_span'] = True
                except Exception as exc:
                    # A boundary-model failure must not erase a completed LLM verdict.
                    state.errors.append(f'fusion_span:{type(exc).__name__}')

    def _fusion_safety_payload(self, state):
        return {'conversation': state.messages, 'candidate_answer': state.candidate_answer}

    def _fusion_safety_system(self, state):
        return self.guard_safety_system

    async def _fusion_safety(self, state):
        state.safe = False
        payload = self._fusion_safety_payload(state)
        try:
            raw = await self.llm.chat(self._fusion_safety_system(state),
                json.dumps(payload, ensure_ascii=False), effort='low', max_tokens=1800,
                timeout=self.guard_safety_timeout, json_mode=True)
            safe = (_json_object(raw) or {}).get('safe')
            if not isinstance(safe, bool):
                raise ValueError('Invalid safety verdict')
            state.safe = safe
        except Exception as exc:
            state.errors.append(f'independent_safety:{type(exc).__name__}')

    async def _fusion_stage(self, coroutine, seconds, state, name):
        try:
            await asyncio.wait_for(coroutine, max(.001, seconds))
            return True
        except asyncio.TimeoutError:
            state.errors.append(f'{name}:Deadline')
        except Exception as exc:
            state.errors.append(f'{name}:{type(exc).__name__}')
        return False

    async def _run_safety(self, safety_state, input_state, input_task, started):
        if self.guard_safety_after_input:
            completed = await asyncio.shield(input_task)
            safety_state.label = input_state.label
            safety_state.debug['fusion_input_reliable'] = bool(completed and not input_state.errors)
        remaining = self.guard_deadline - (time.monotonic() - started)
        if remaining <= 0:
            safety_state.errors.append('safety:NoBudget')
            return
        await self._fusion_stage(self._fusion_safety(safety_state),
            min(self.guard_safety_timeout, remaining), safety_state, 'safety')

    async def guard(self, body):
        started = time.monotonic()
        state = self._new_state(body, True)
        ground_state = safety_state = None

        async def work():
            nonlocal ground_state, safety_state
            async with self._slots:
                prepare_started = time.monotonic()
                remaining = self.guard_deadline - (prepare_started - started)
                if not await self._fusion_stage(self._search(state, extra=state.candidate_answer or ''),
                        min(self.guard_prepare_limit, remaining), state, 'search'):
                    return
                state.section = state.chunks[0].section_id if state.chunks else None
                remaining = self.guard_deadline - (time.monotonic() - started)
                if remaining <= 0:
                    state.errors.append('request:Deadline')
                    return
                prepare_left = self.guard_prepare_limit - (time.monotonic() - prepare_started)
                tasks = []
                try:
                    if state.has_candidate:
                        # Snapshot before classification can mutate label/section.
                        # Guard output retains the original first search section.
                        ground_state, safety_state = isolated_state(state), isolated_state(state)
                        safety_state.safe = False
                    input_task = asyncio.create_task(self._fusion_stage(
                        self._classify(state), min(prepare_left, remaining), state, 'prepare'))
                    tasks.append(input_task)
                    if state.has_candidate:
                        skip_ground = False
                        if self.guard_fast_input_wait > 0:
                            await asyncio.wait({input_task},
                                timeout=min(self.guard_fast_input_wait, remaining))
                            skip_ground = (input_task.done() and not input_task.cancelled()
                                and input_task.exception() is None and input_task.result() is True
                                and not state.errors and state.label is not None
                                and state.decision is Decision.BLOCK)
                        remaining = self.guard_deadline - (time.monotonic() - started)
                        if remaining <= 0:
                            state.errors.append('request:Deadline')
                            return
                        if skip_ground:
                            ground_state.debug['fusion_ground_skipped_fast_block'] = True
                        else:
                            tasks.append(asyncio.create_task(self._fusion_stage(
                                self._fusion_ground(ground_state),
                                min(self.guard_ground_timeout, remaining), ground_state, 'ground')))
                        tasks.append(asyncio.create_task(
                            self._run_safety(safety_state, state, input_task, started)))
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await asyncio.wait_for(work(), self.guard_deadline)
        except asyncio.TimeoutError:
            state.errors.append('request:Deadline')
        finally:
            if state.has_candidate and ground_state is not None and safety_state is not None:
                state.grounded, state.unsupported = ground_state.grounded, ground_state.unsupported
                if state.decision is Decision.BLOCK:
                    state.grounded, state.unsupported = True, []
                state.safe = safety_state.safe
                state.errors.extend(ground_state.errors)
                state.errors.extend(safety_state.errors)
                state.debug['fusion_completed_ground_votes'] = ground_state.debug.get('fusion_completed_ground_votes', 0)
                if ground_state.debug.get('fusion_ground_skipped_fast_block'):
                    state.debug['fusion_ground_skipped_fast_block'] = True
                if 'fusion_completed_safety_votes' in safety_state.debug:
                    state.debug['fusion_completed_safety_votes'] = safety_state.debug['fusion_completed_safety_votes']
                if ground_state.debug.get('learned_span'):
                    state.debug['learned_span'] = True
            elif state.has_candidate:
                state.safe = False
        return state
