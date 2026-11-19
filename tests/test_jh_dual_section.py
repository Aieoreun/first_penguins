import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import State
from pipelines.kimjaehak.dual_section import ContextFast, DualQuery, Pipeline


class DualSectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_confident_section_is_not_overridden_by_dual_generation(self):
        p = Pipeline.__new__(Pipeline)
        s = State(messages=[], label=Label.NORMAL, section='chosen', debug={'tutor_section_gate': .92})
        with patch.object(ContextFast, '_generate', new_callable=AsyncMock) as original, patch.object(DualQuery, '_generate', new_callable=AsyncMock) as dual:
            await p._generate(s)
            original.assert_awaited_once_with(p, s)
            dual.assert_not_awaited()

    async def test_uncertain_sources_use_dual_query(self):
        p = Pipeline.__new__(Pipeline)
        s = State(messages=[], label=Label.NORMAL)
        with patch.object(DualQuery, '_generate', new_callable=AsyncMock) as dual:
            await p._generate(s)
            dual.assert_awaited_once_with(p, s)

    def test_both_guard_markers_and_deadline_survive_composition(self):
        from pipelines import load
        self.assertIs(load('jh_r3_dual_section90'), Pipeline)
        p = Pipeline.__new__(Pipeline)
        s = p._new_state({'messages': [{'role':'user','content':'질문'}]}, True)
        self.assertTrue(s._dual_guard)
        self.assertTrue(s._section_gate_guard)
        self.assertGreater(s.round2_deadline, 0)
        for method in ('guard', '_check_output'):
            self.assertIs(getattr(Pipeline, method), getattr(ContextFast, method))
