"""Ablate reasoning effort only for uncertain tutor source selection."""
import json
from .focused_excerpt import Selected as SelectedFocused


class Pipeline(SelectedFocused):
    source_minimal_effort = True

    async def _source_response(self, state, ids, timeout):
        return await self.llm.chat(self.source_extract_system,
            json.dumps(self._source_payload(state, ids), ensure_ascii=False),
            effort='minimal', max_tokens=2000, timeout=timeout, json_mode=True)
