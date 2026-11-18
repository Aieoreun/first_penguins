"""Expose successful resolved retrieval text to the original tutor material check."""
import json

from .round6_query_hint import HINT_RULE, SourceHint


class Pipeline(SourceHint):
    tutor_material_resolved_hint = True

    async def _ask(self, state, name, system, user, effort, timeout):
        hint = getattr(state, '_resolved_source_hint', None)
        if (name == 'material' and not getattr(state, '_material_budget_guard', True)
                and isinstance(hint, str) and hint):
            system = system + HINT_RULE
            user = user + '\n\n[검색 보조 데이터(JSON)]\n' + json.dumps(
                {'retrieval_query': hint}, ensure_ascii=False)
        return await super()._ask(state, name, system, user, effort, timeout)
