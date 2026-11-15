"""Use multiple answer sources only when the learned section gate abstains."""
from core.schema import Label
from .candidate import Pipeline as Candidate
from .multiground import Pipeline as MultiGround


class Pipeline(Candidate):
    answer_sections = 3
    uncertain_sources_only = True

    async def _generate(self, state):
        if (state.label in {Label.NORMAL, Label.PII_DISCLOSURE}
                and not state.debug.get("learned_section")
                and len({chunk.section_id for chunk in state.chunks}) > 1):
            return await MultiGround._generate(self, state)
        return await super()._generate(state)
