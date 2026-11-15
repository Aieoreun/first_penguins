"""Refine an LLM-identified error span with a confident scratch boundary model."""
from .context_policy import Pipeline as ContextPolicy
from .span_tagger import SpanTagger, public_rows, refine_spans
from .voted_output import Five


class Pipeline(ContextPolicy):
    learned_span_c = 100
    learned_span_threshold = .95
    training_source = "bundled_public_categories_evidence_harmful_actions_and_output_spans"

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self.span_tagger = SpanTagger(public_rows(), self.learned_span_c)

    async def _check_output(self, state):
        await super()._check_output(state)
        if not state.unsupported:
            return
        predicted, confidence = self.span_tagger.predict(state.candidate_answer or "")
        spans, changed = refine_spans(state.candidate_answer or "", state.unsupported,
                                     predicted, confidence, self.learned_span_threshold)
        if changed:
            state.unsupported = spans
            state.debug["learned_span"] = True
        # Neither grounded nor safe is changed by this boundary refinement.


class Consensus(Pipeline, Five):
    pass
