"""Use extra reasoning only when two cheap input votes disagree."""
from pipelines.v1.pipeline import _json_object
from .adaptive import Pipeline as Adaptive
from .calibrated import BLOCK_SYSTEM
from .reasoned import stream_text


class Pipeline(Adaptive):
    tie_effort = "medium"
    tie_max_tokens = 4000

    async def _tie_vote(self, state, name, system, check, remaining):
        if name == "block":
            system = BLOCK_SYSTEM
        try:
            text = await stream_text(self.llm, system, check, effort=self.tie_effort,
                                     max_tokens=self.tie_max_tokens, timeout=remaining)
            obj = _json_object(text)
            if obj is None:
                state.errors.append(f"{name}_tie:parse")
            return obj or {}
        except Exception as exc:
            state.errors.append(f"{name}_tie:{type(exc).__name__}")
            return {}
