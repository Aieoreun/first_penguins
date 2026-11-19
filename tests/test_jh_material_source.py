import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import Chunk, State
from pipelines.kimjaehak.material_source import ContextFast, Pipeline


class MaterialSourceTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self):
        p = Pipeline.__new__(Pipeline)
        p.searcher = SimpleNamespace(section_text=lambda sid:'추가 납입은 매달 10만 원까지 가능하다.')
        s = State(messages=[], label=Label.NORMAL, section='old', chunks=[Chunk('new','source')])
        s._material_source_result = {'section':'new','quote':'추가 납입은 매달 10만 원까지 가능하다.'}
        return p,s

    async def test_valid_existing_material_quote_changes_source_only(self):
        p,s = self.setup_case()
        with patch.object(ContextFast, '_classify', new_callable=AsyncMock):
            await p._classify(s)
        self.assertEqual(s.section,'new')
        self.assertEqual(s.label,Label.NORMAL)

    async def test_guard_policy_confidence_and_invalid_quotes_preserve_source(self):
        cases=[{'guard':True},{'label':Label.OUT_OF_MATERIAL},{'label':Label.HARMFUL_REQUEST},{'confident':True},
               {'quote':'추가 납입은 매달 20만 원까지 가능하다.'},{'quote':'10'},{'section':'unretrieved'}]
        for case in cases:
            p,s = self.setup_case()
            s._material_source_guard = case.get('guard',False)
            s.label = case.get('label',Label.NORMAL)
            s.debug['learned_section'] = case.get('confident',False)
            s._material_source_result.update({k:v for k,v in case.items() if k in {'section','quote'}})
            with patch.object(ContextFast, '_classify', new_callable=AsyncMock):
                await p._classify(s)
            self.assertEqual(s.section,'old',case)

    async def test_capture_does_not_change_call_or_guard_prediction(self):
        p,s = self.setup_case()
        s._material_source_result = None
        prediction={'section':'new','quote':'추가 납입은 매달 10만 원까지 가능하다.'}
        with patch.object(ContextFast, '_ask', new_callable=AsyncMock,return_value=prediction) as original:
            output=await p._ask(s,'material','sys','user','low',10)
            original.assert_awaited_once_with(s,'material','sys','user','low',10)
        self.assertIs(output,prediction)
        self.assertIs(s._material_source_result,prediction)
        s._material_source_guard, s._material_source_result = True,None
        with patch.object(ContextFast, '_ask', new_callable=AsyncMock,return_value=prediction):
            await p._ask(s,'material','sys','user','low',10)
        self.assertIsNone(s._material_source_result)

    def test_registry_and_guard_output_match(self):
        from pipelines import load
        self.assertIs(load('jh_r3_material_source'),Pipeline)
        for method in ('guard','_generate','_search','_check_output'):
            self.assertIs(getattr(Pipeline,method),getattr(ContextFast,method))
