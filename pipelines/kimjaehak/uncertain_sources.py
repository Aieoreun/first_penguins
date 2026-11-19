"""Use multiple answer sources only when the learned section gate abstains."""
from core.schema import Label
from .candidate import Pipeline as Candidate
from .multiground import Pipeline as MultiGround


class Pipeline(Candidate):
    answer_sections = 3
    uncertain_sources_only = True
    retry_source_failure = False

    async def _generate(self, state):
        if (state.label in {Label.NORMAL, Label.PII_DISCLOSURE}
                and not state.debug.get("learned_section")
                and len({chunk.section_id for chunk in state.chunks}) > 1):
            original_section, error_count = state.section, len(state.errors)
            await MultiGround._generate(self, state)
            if self.retry_source_failure and len(state.errors) > error_count:
                state.section = original_section
                return await super()._generate(state)
            return
        return await super()._generate(state)


class Resilient(Pipeline):
    retry_source_failure = True


class Six(Resilient):
    answer_sections = 6
