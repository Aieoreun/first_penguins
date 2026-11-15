"""Keep v4 grounding semantics, but give the output check more reasoning room."""
from core.schema import Decision
from core.spans import locate_all
from pipelines.v1 import prompts
from pipelines.v1.pipeline import SECOND_TIMEOUT, _json_object
from .reasoned import stream_text
from .voted import Pipeline as Voted


class Pipeline(Voted):
    output_effort = "medium"
    output_max_tokens = 4000

    async def _check_output(self, state):
        if state.decision is Decision.BLOCK:
            return await super()._check_output(state)
        state.safe = True
        candidate = state.candidate_answer or ""
        material = self.searcher.section_text(state.section) if state.section else "\n\n".join(c.text for c in state.chunks[:5])
        try:
            text = await stream_text(self.llm, self.unsupported_system, prompts.unsupported_user(material, candidate),
                                     effort=self.output_effort, max_tokens=self.output_max_tokens, timeout=SECOND_TIMEOUT)
            phrases = (_json_object(text) or {}).get("unsupported")
            if not isinstance(phrases, list):
                raise ValueError("Invalid unsupported phrases")
            state.unsupported = locate_all(candidate, phrases)
        except Exception as exc:
            state.errors.append(f"medium_grounding:{type(exc).__name__}")
            state.unsupported = []
        state.grounded = not state.unsupported
