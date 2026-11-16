"""Keep trusted tutor section predictions before dual-query source selection."""
from .dual_query import Pipeline as DualQuery
from .guard_fitted import ContextFast
from .tutor_section_gate import Pipeline as SectionGate


class Pipeline(SectionGate, DualQuery):
    async def _generate(self, state):
        if state.debug.get('tutor_section_gate'):
            return await ContextFast._generate(self, state)
        return await DualQuery._generate(self, state)
