import unittest
from unittest.mock import AsyncMock, patch

from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.tutor_boundaries import Material, Other, MATERIAL_BOUNDARY, OTHER_BOUNDARY


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_selected_tutor_check_changes(self):
        for cls, selected, addition in [(Other, 'other', OTHER_BOUNDARY),
                                        (Material, 'material', MATERIAL_BOUNDARY)]:
            p = cls.__new__(cls)
            for guard in (False, True):
                state = p._new_state({'messages': []}, guard)
                # Upstream classification replaces debug; isolation must survive.
                state.debug = {}
                for name in ('block', 'other', 'material'):
                    with patch.object(ContextFast, '_ask', new_callable=AsyncMock,
                                      return_value={'category': 'none'}) as original:
                        result = await p._ask(state, name, 'original', 'payload', 'low', 10)
                    expected = 'original' + (addition if not guard and name == selected else '')
                    original.assert_awaited_once_with(state, name, expected, 'payload', 'low', 10)
                    self.assertEqual(result, {'category': 'none'})

    def test_guard_generation_and_budget_are_inherited(self):
        from pipelines import load
        for name, cls in [('jh_r3_tutor_other_boundary', Other),
                          ('jh_r3_tutor_material_boundary', Material)]:
            self.assertIs(load(name), cls)
            for method in ('guard', '_generate', '_check_output', '_collect_checks'):
                self.assertIs(getattr(cls, method), getattr(ContextFast, method))
            self.assertEqual(cls.deadline, ContextFast.deadline)


if __name__ == '__main__':
    unittest.main()
