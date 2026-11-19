"""Race a fast grounding fallback with a deadline-bounded reasoning check."""
import asyncio
from dataclasses import replace
import json
import time

from core.schema import Decision
from core.spans import locate_all
from pipelines.v1.pipeline import _json_object
from .context_grounding import Pipeline as ContextGrounding
from .reasoned import stream_text
from .span_tagger import refine_spans


class Pipeline(ContextGrounding):
    hybrid_output = True
    output_effort = "medium"
    output_max_tokens = 4000
    output_timeout = 20

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        # Only this variant uses the private request-local timing attribute.
        state._ground_deadline = time.monotonic() + self.deadline - .5
        return state

    async def _check_output(self, state):
        if state.decision is Decision.BLOCK:
            return await super()._check_output(state)
        budget = max(.01, min(self.output_timeout,
            getattr(state, "_ground_deadline", time.monotonic()+self.output_timeout)-time.monotonic()))
        low = replace(state, errors=[], unsupported=[], debug=dict(state.debug))
        async def fast():
            try:
                await asyncio.wait_for(super(Pipeline, self)._check_output(low), budget)
            except TimeoutError:
                low.errors.append("hybrid_fast:TimeoutError")
        async def reasoned():
            try:
                raw = await asyncio.wait_for(stream_text(self.llm, self.unsupported_system,
                    json.dumps(self._ground_payload(state),ensure_ascii=False),effort=self.output_effort,
                    max_tokens=self.output_max_tokens,timeout=budget),budget)
                phrases = (_json_object(raw) or {}).get("unsupported")
                if not isinstance(phrases,list):raise ValueError("Invalid unsupported phrases")
                spans = locate_all(state.candidate_answer or "", phrases)
                if spans:
                    predicted,confidence = self.span_tagger.predict(state.candidate_answer or "")
                    spans,_ = refine_spans(state.candidate_answer or "",spans,predicted,confidence,self.learned_span_threshold)
                return spans,None
            except Exception as exc:
                return None,f"hybrid_reasoning:{type(exc).__name__}"
        _,(spans,error) = await asyncio.gather(fast(),reasoned())
        state.errors.extend(low.errors)
        if error:
            state.errors.append(error)
        state.unsupported = low.unsupported if error else spans
        state.grounded, state.safe = not state.unsupported, True
        state.debug["hybrid_reasoning_used"] = error is None
