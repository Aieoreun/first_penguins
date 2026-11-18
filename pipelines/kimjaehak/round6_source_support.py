"""Check support only when a trusted normal input still needs source selection."""
from copy import copy
import time

from core.schema import Label
from pipelines.v1.pipeline import _json_object
from .focused_excerpt import Pipeline as FocusedExcerpt
from .material_budget import Pipeline as MaterialBudget
from .round6_joint_quote import QUOTE_SYSTEM


class Pipeline(MaterialBudget):
    tutor_source_support = True
    source_support_system = QUOTE_SYSTEM
    # The inherited non-stream selector uses low effort and exactly 2000 tokens.
    source_support_tokens = 2000
    source_support_timeout = 8

    async def _generate(self, state):
        if (getattr(state, '_material_budget_guard', True)
                or not state.section or state.label is not Label.NORMAL
                or not state.debug.get('learned_gate')
                or state.debug.get('learned_section') or state.debug.get('tutor_section_gate')):
            return await super()._generate(state)
        try:
            ids = await self._candidate_sections(state)
            remaining = getattr(state, '_source_extract_deadline', time.monotonic()+13)-time.monotonic()-.5
            if len(ids) > 1 and remaining >= 1:
                view = copy(self)
                view.source_extract_system = self.source_support_system
                raw = await view._source_response(state, ids,
                    timeout=min(self.source_support_timeout, remaining))
                result = _json_object(raw) or {}
                supported, selected = result.get('supported'), result.get('section')
                if (not isinstance(supported, bool) or not isinstance(selected, str)
                        or selected not in ids):
                    raise ValueError('Invalid source support decision')
                # asked_fact/quote guide the model only. They are neither
                # lexically verified here nor used as the student's answer.
                state.debug['source_support_supported'] = supported
                if not supported:
                    state.label = Label.OUT_OF_MATERIAL
                    # The inherited PlainAbstain branch is fixed text: no API.
                    return await super()._generate(state)
                state.section = selected
                state.debug['source_support_selected'] = True
        except Exception as exc:
            state.errors.append(f'source_support:{type(exc).__name__}')
        # No second selector after failure. Preserve the excerpt and its
        # inherited empty/over-4000-character generation fallback.
        return await FocusedExcerpt._generate(self, state)


class Budget(Pipeline):
    """Extend only support checks; keep the ordinary source selector at 8s."""
    source_support_timeout = 12
