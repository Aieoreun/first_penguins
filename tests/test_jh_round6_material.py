import unittest
from unittest.mock import AsyncMock, patch

from pipelines import load
from pipelines.v2 import prompts
from pipelines.kimjaehak.material_budget import Pipeline as Base
from pipelines.kimjaehak.resolved_query import TopicSelect
from pipelines.kimjaehak.round6_material import COMPUTABLE_SUPPORT


class MaterialTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_tutor_material_instruction_changes(self):
        cls=load('jh_r6_computable_material');p=cls.__new__(cls)
        for guard in (False,True):
            state=p._new_state({'messages':[{'role':'user','content':'code question'}]},guard)
            for name in ('material','block','other'):
                mock=AsyncMock(return_value={'quote':'evidence'})
                with patch.object(Base,'_ask',mock):
                    result=await p._ask(state,name,prompts.MATERIAL_SYSTEM,'original', 'low',10)
                args=mock.call_args.args
                expected=prompts.MATERIAL_SYSTEM+(COMPUTABLE_SUPPORT if name=='material' and not guard else '')
                self.assertEqual(args,(state,name,expected,'original','low',10))
                self.assertEqual(result,{'quote':'evidence'})
        self.assertNotIn('# 코드와 표에 들어 있는 근거',prompts.MATERIAL_SYSTEM)

    def test_topic_selector_uses_existing_choice_and_focused_answer(self):
        cls=load('jh_r6_topic_focused')
        self.assertIs(cls._resolve_query,TopicSelect._resolve_query)
        self.assertIs(cls._generate,Base._generate)
        self.assertIs(cls._source_response,Base._source_response)
        self.assertEqual((cls.query_resolve_timeout,cls.query_resolve_effort),(4,'minimal'))

    def test_guard_and_material_budgets_unchanged(self):
        for name in ('jh_r6_topic_focused','jh_r6_computable_material'):
            cls=load(name)
            self.assertIs(cls.guard,Base.guard)
            self.assertIs(cls._check_output,Base._check_output)
            self.assertEqual((cls.deadline,cls.tutor_prepare_limit,cls.tutor_material_timeout),(26,16,13))


if __name__=='__main__': unittest.main()
