"""Spend remaining request time on ordinary answer reasoning, not classification."""
from core.schema import Label
from pipelines.v1 import prompts
from pipelines.v1.pipeline import _SECTION_MARK
from .candidate import Pipeline as Candidate
from .reasoned import stream_text


class Pipeline(Candidate):
    stream_answer_effort = "medium"
    stream_answer_max_tokens = 4000
    stream_answer_timeout = 22

    async def _generate(self, state):
        if state.label not in {Label.NORMAL, Label.PII_DISCLOSURE} or not state.section:
            return await super()._generate(state)
        try:
            text = await stream_text(
                self.llm, self.answer_systems[state.label],
                prompts.answer_user(self.searcher.section_text(state.section), state.history, state.last_user),
                effort=self.stream_answer_effort, max_tokens=self.stream_answer_max_tokens,
                timeout=self.stream_answer_timeout, json_mode=False)
            text = _SECTION_MARK.sub("", text).strip()
            if not text:
                raise ValueError("Empty answer")
            state.answer, state.citations = f"{text} ({state.section})", [state.section]
        except Exception as exc:
            # The inherited 26-second request deadline also bounds this fallback.
            state.errors.append(f"medium_answer:{type(exc).__name__}")
            await super()._generate(state)
