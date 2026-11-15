"""Keep the original grounding rubric, add conversation and bounded KB context."""
import json

from core.schema import Decision
from core.spans import locate_all
from pipelines.v1.pipeline import GUARD_EFFORT, SECOND_TIMEOUT, _json_object
from .learned_sections import TutorOnly


class Pipeline(TutorOnly):
    output_sections = 1

    async def _check_output(self, state):
        if state.decision is Decision.BLOCK:
            return await super()._check_output(state)
        state.safe = True
        ids = ([state.section] if state.section else []) + [chunk.section_id for chunk in state.chunks]
        ids = list(dict.fromkeys(ids))[:self.output_sections]
        candidate = state.candidate_answer or ""
        payload = {"교재": [{"section": sid, "text": self.searcher.section_text(sid)} for sid in ids],
                   "대화": state.messages, "후보 답변": candidate}
        try:
            raw = await self.llm.chat(self.unsupported_system, json.dumps(payload, ensure_ascii=False),
                                      effort=GUARD_EFFORT, max_tokens=1500,
                                      timeout=SECOND_TIMEOUT, json_mode=True)
            phrases = (_json_object(raw) or {}).get("unsupported")
            if not isinstance(phrases, list):
                raise ValueError("Invalid unsupported phrases")
            state.unsupported = locate_all(candidate, phrases)
        except Exception as exc:
            state.errors.append(f"context_grounding:{type(exc).__name__}")
            state.unsupported = []
        state.grounded = not state.unsupported


class ThreeSections(Pipeline):
    output_sections = 3
