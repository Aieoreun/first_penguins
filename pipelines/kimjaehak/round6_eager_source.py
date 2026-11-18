"""Prefetch the unchanged tutor source selector beside original input checks."""
import asyncio
from copy import copy
import time

from core.schema import Label, parse_label
from pipelines.v1.pipeline import _json_object
from .focused_excerpt import Pipeline as FocusedExcerpt
from .material_budget import Pipeline as MaterialBudget


class Eager(MaterialBudget):
    tutor_eager_source = True

    async def _prefetch_source(self, view, ids, timeout):
        try:
            raw = await asyncio.wait_for(self._source_response(view, ids, timeout), timeout)
            return self._validate_source(_json_object(raw) or {}, ids), None
        except Exception as exc:
            return None, f'source_extract:{type(exc).__name__}'

    def _start_prefetch(self, state):
        if not state.chunks or not state.chunks[0].section_id:
            return
        probability = self.classifier.predict_proba(
            self.vectorizer.transform([state.last_user]))[0]
        index = probability.argmax()
        label = parse_label(str(self.classifier.classes_[index]))
        if (label in {Label.NORMAL, Label.PII_DISCLOSURE} and probability[index] >= .5
                and not getattr(state, '_eager_source_attempted', False)):
            view = copy(state)
            view.section = state.chunks[0].section_id if state.chunks else None
            view.debug, view.errors = dict(state.debug), list(state.errors)
            ids = self._source_candidates(view)
            remaining = getattr(state, '_source_extract_deadline', time.monotonic()) - time.monotonic() - .5
            if len(ids) > 1 and remaining >= 1:
                state._eager_source_task = asyncio.create_task(self._prefetch_source(
                    view, ids, min(self.source_extract_timeout, remaining)))
                state._eager_source_attempted = True

    async def _collect_checks(self, state, check, material):
        if not getattr(state, '_material_budget_guard', True):
            try:
                self._start_prefetch(state)
            except Exception as exc:
                state.errors.append(f'source_prefetch_setup:{type(exc).__name__}')
        # In particular, retain the original chunk material payload and votes.
        return await super()._collect_checks(state, check, material)

    async def _classify(self, state):
        if getattr(state, '_material_budget_guard', True):
            return await super()._classify(state)
        state._eager_source_task = None
        state._eager_source_attempted = False
        state._eager_classification_complete = False
        state._eager_source_reused = False
        selected = None
        eligible = False
        try:
            await super()._classify(state)
            eligible = (state.label in {Label.NORMAL, Label.PII_DISCLOSURE}
                        and not state.debug.get('learned_section')
                        and not state.debug.get('tutor_section_gate'))
            task = state._eager_source_task
            if task is not None and eligible:
                selected, _ = await task
        finally:
            task = state._eager_source_task
            if task is not None:
                if not task.done():
                    task.cancel()
                result = (await asyncio.gather(task, return_exceptions=True))[0]
                if isinstance(result, tuple) and result[1]:
                    state.errors.append(result[1])
            state._eager_source_task = None
        # A prepare cancellation must never commit the prefetched selection.
        state._eager_classification_complete = True
        if eligible and state._eager_source_attempted:
            if selected is not None:
                state.section = selected
                state._eager_source_reused = True
                state.debug['eager_source_selected'] = True

    async def _generate(self, state):
        if (not getattr(state, '_material_budget_guard', True)
                and getattr(state, '_eager_source_attempted', False)
                and state.label in {Label.NORMAL, Label.PII_DISCLOSURE}
                and not state.debug.get('learned_section')
                and not state.debug.get('tutor_section_gate')):
            # At most one selector attempt, even after a prepare cancellation.
            # Keep the parent's section/label and excerpt/no-section fallbacks.
            return await FocusedExcerpt._generate(self, state)
        return await super()._generate(state)
