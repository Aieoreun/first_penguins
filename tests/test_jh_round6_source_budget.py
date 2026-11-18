import unittest
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.schema import Label
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.round6_source_budget import Budget, Reason


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_source_call_parameters_change(self):
        for cls,effort,tokens in ((Budget,'low',2000),(Reason,'medium',2000)):
            p=cls.__new__(cls)
            p.llm=SimpleNamespace(chat=AsyncMock(return_value='{"section":"A"}'))
            p.searcher=SimpleNamespace(section_text=lambda sid:'original')
            state=SimpleNamespace(messages=[])
            await p._source_response(state,['A'],2.5)
            self.assertEqual(p.llm.chat.call_args.kwargs,
                {'effort':effort,'max_tokens':tokens,'timeout':2.5,'json_mode':True})
            self.assertIs(cls.guard,Base.guard)
            self.assertIs(cls._check_output,Base._check_output)

    async def test_parent_clamps_source_to_remaining_request_budget(self):
        p=Reason.__new__(Reason)
        p._candidate_sections=AsyncMock(return_value=['A','B'])
        p._source_response=AsyncMock(return_value='{"section":"B"}')
        p._excerpt=lambda state:'real evidence'
        state=SimpleNamespace(section='A',label=Label.NORMAL,debug={},errors=[],
            _extract_guard=False,_source_extract_deadline=time.monotonic()+2)
        await p._generate(state)
        self.assertLessEqual(p._source_response.call_args.kwargs['timeout'],1.5)
        self.assertEqual(state.citations,['B'])
        self.assertEqual(p.deadline,26)


if __name__=='__main__': unittest.main()
