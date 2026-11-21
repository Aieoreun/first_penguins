"""Opt-in learned span veto after completed B guard output, not an ID lookup."""
import time
import numpy as np
from core.schema import Decision
from .round7_binary_guard import BinaryGuard


class SpanVeto(BinaryGuard):
    guard_span_veto_threshold=.95
    guard_span_veto_max_chars=4000

    async def _fusion_post_checks(self,input_state,state,started):
        await super()._fusion_post_checks(input_state,state,started)
        if (state is None or not state.has_candidate or input_state.decision is Decision.BLOCK
                or not state.grounded or len(state.candidate_answer)>self.guard_span_veto_max_chars
                or time.monotonic()-started>self.guard_deadline-.1):
            return
        try:
            spans,confidence=self.span_tagger.predict(state.candidate_answer)
            if not np.isfinite(confidence) or confidence<self.guard_span_veto_threshold or not spans:return
            if not isinstance(spans,list) or any(not isinstance(s,str) or not s or s not in state.candidate_answer for s in spans):
                raise ValueError('Invalid learned veto spans')
            state.unsupported=spans;state.grounded=False
            input_state.debug['r8_span_veto']=True
        except Exception as exc:
            state.errors.append('span_veto:'+type(exc).__name__)
        return
