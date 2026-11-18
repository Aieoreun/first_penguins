"""Tutor source ablations: evidence order and bounded parallel selection."""
import asyncio
from collections import Counter
from copy import copy
import json
import time

from pipelines.v1.pipeline import _json_object
from .material_budget import Pipeline as MaterialBudget


class EvidenceFirst(MaterialBudget):
    source_evidence_order = 'focused_subsections'

    def _source_payload(self, state, ids):
        # Preserve every subsection, code fence and table; reorder only. The
        # original state and final answer path are untouched.
        view = copy(state)
        materials = []
        for sid in ids:
            view.section = sid
            materials.append({'section': sid, 'text': self._excerpt(view)})
        return {'conversation': state.messages, 'materials': materials}


class VoteThree(MaterialBudget):
    source_selection_votes = 3

    async def _source_response(self, state, ids, timeout):
        # The whole group shares the existing eight-second source budget.
        # Two matching valid decisions can finish early; a three-way tie uses
        # call order, never request completion order.
        deadline = time.monotonic() + timeout
        parent = super()._source_response
        tasks = [asyncio.create_task(parent(state, ids, timeout)) for _ in range(3)]
        pending = set(tasks)
        valid = {}
        try:
            while pending:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                done, pending = await asyncio.wait(pending, timeout=left,
                                                   return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    break
                for task in sorted(done, key=tasks.index):
                    try:
                        parsed = _json_object(task.result()) or {}
                        valid[tasks.index(task)] = self._validate_source(parsed, ids)
                    except Exception as exc:
                        state.errors.append(f'source_vote:{type(exc).__name__}')
                counts = Counter(valid.values())
                if counts and max(counts.values()) >= 2:
                    break
            for task in pending:
                # Deliberately cancelled redundant votes are not failures.
                if not any(v >= 2 for v in Counter(valid.values()).values()):
                    state.errors.append('source_vote:TimeoutError')
            if not valid:
                raise ValueError('No valid source vote')
            ordered = [valid[i] for i in sorted(valid)]
            counts = Counter(ordered)
            selected = max(ordered, key=counts.get)
            state.debug['source_votes_valid'] = len(valid)
            state.debug['source_votes_agreement'] = counts[selected]
            return json.dumps({'section': selected})
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
