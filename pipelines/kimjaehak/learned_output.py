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


class Stronger(Consensus):
    learned_gate_c = 3000


class GuardCategory(Stronger):
    guard_category_without_subtype = True

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state.debug["guard_endpoint"] = guard
        return state

    async def _classify(self, state):
        # /guard exposes harmful_content, not the tutor's request/speech action.
        if state.debug.get("guard_endpoint"):
            probability = self.classifier.predict_proba(self.vectorizer.transform([state.last_user]))[0]
            index = probability.argmax()
            if self.classifier.classes_[index] == "harmful_content" and probability[index] >= self.learned_gate_threshold:
                from core.schema import Label
                state.label = Label.HARMFUL_REQUEST
                state.section = state.chunks[0].section_id if state.chunks else None
                state.debug = {"learned_guard_category":True,"confidence":float(probability[index])}
                return
        return await super()._classify(state)
