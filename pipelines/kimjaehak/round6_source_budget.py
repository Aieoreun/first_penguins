"""Use remaining tutor time for non-streaming source selection."""
import json

from .material_budget import Pipeline as MaterialBudget


class Budget(MaterialBudget):
    source_extract_timeout = 12
    source_selection_effort = 'low'
    source_selection_tokens = 2000

    async def _source_response(self, state, ids, timeout):
        # SelectedSection computes timeout from the unchanged request deadline.
        return await self.llm.chat(self.source_extract_system,
            json.dumps(self._source_payload(state, ids), ensure_ascii=False),
            effort=self.source_selection_effort, max_tokens=self.source_selection_tokens,
            timeout=timeout, json_mode=True)


class Reason(Budget):
    source_selection_effort = 'medium'
    source_selection_tokens = 3000
