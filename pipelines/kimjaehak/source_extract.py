"""Spend the tutor generation call on source selection, then quote the KB."""
import json
import time

from core.schema import Label
from pipelines.v1.pipeline import _json_object
from .extractive_answer import Section
from .material_source import Pipeline as MaterialSource
from .source_choice import SYSTEM as SOURCE_SYSTEM
from .tutor_section_gate import Pipeline as SectionGate


class MaterialSection(MaterialSource, Section):
    pass


class GatedSection(SectionGate, Section):
    pass


class SelectedSection(GatedSection):
    source_extraction_selector = True
    source_extract_timeout = 8
    source_extract_sections = 6

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._source_extract_deadline = time.monotonic() + self.deadline
        return state

    async def _generate(self, state):
        if (getattr(state, '_extract_guard', False) or not state.section
                or state.label not in {Label.NORMAL, Label.PII_DISCLOSURE}
                or state.debug.get('learned_section') or state.debug.get('tutor_section_gate')):
            return await super()._generate(state)
        last = self.searcher.search(state.last_user, bm25_top=1, embedding_top=0)
        ids = list(dict.fromkeys([state.section] + [c.section_id for c in last]
                                + [c.section_id for c in state.chunks]))[:self.source_extract_sections]
        remaining = getattr(state, '_source_extract_deadline', time.monotonic()+13)-time.monotonic()-.5
        if len(ids) > 1 and remaining >= 1:
            try:
                raw = await self.llm.chat(SOURCE_SYSTEM,
                    json.dumps({'conversation':state.messages,
                                'materials':[{'section':sid,'text':self.searcher.section_text(sid)} for sid in ids]},
                               ensure_ascii=False), effort='low', max_tokens=2000,
                    timeout=min(self.source_extract_timeout,remaining),json_mode=True)
                selected = (_json_object(raw) or {}).get('section')
                if selected not in ids:
                    raise ValueError('Invalid source selection')
                state.section = selected
                state.debug['source_extraction_selected'] = True
            except Exception as exc:
                state.errors.append(f'source_extract:{type(exc).__name__}')
        # Deterministic excerpt is also the fallback; no speculative second LLM call.
        return await super()._generate(state)
