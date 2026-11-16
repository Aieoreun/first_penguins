"""Apply semantic examples only to /tutor; preserve the complete guard path."""
from .guard_fitted import ContextFast
from .input_examples import DenseTen


class Pipeline(ContextFast):
    tutor_type = DenseTen
    endpoint_variants = {"tutor": "jh_r2_dense10", "guard": "jh_guard_fitted_contextfast"}

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self._tutor_pipeline = self.tutor_type(searcher, llm)
        # Both endpoints consume the same request capacity. Models/configuration
        # stay separate and immutable; no request-time attribute swapping.
        self._tutor_pipeline._slots = self._slots
        self._tutor_pipeline.request_slots = self.request_slots

    async def tutor(self, body):
        return await self._tutor_pipeline.tutor(body)

    # guard() and all its helpers are inherited unchanged from ContextFast,
    # including guard requests that have no candidate_answer.
