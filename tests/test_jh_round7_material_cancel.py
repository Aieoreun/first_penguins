import asyncio
import itertools
import unittest
from unittest.mock import AsyncMock,patch
from pipelines import load
load('jh_cm_char_mlp50')
from core.schema import Label
from pipelines.v2.pipeline import _priority
from pipelines.kimjaehak.classifier_matrix import CharMLP50
from pipelines.kimjaehak.round7_material_cancel import (
    material_irrelevant,CancelGuardMaterial,CancelTutorMaterial,CancelBothMaterial)
from pipelines.kimjaehak.voted import Pipeline as Voted,BLOCKS,OTHERS
from tests import test_jh_reasoning_ablation as reasoning


class MaterialCancelTests(unittest.IsolatedAsyncioTestCase):
    pipeline=reasoning.ReasoningTests.pipeline
    state=reasoning.ReasoningTests.state

    def test_short_circuit_respects_both_policy_orders_and_minimum_votes(self):
        for block,other,personal,seek in itertools.product(BLOCKS,OTHERS,(True,False),(True,False)):
            blocks=[{'category':block}]*2;others=[{'category':other,'self_info':personal}]*2
            irrelevant=material_irrelevant(blocks,others,1)
            def chosen(missing):
                flags={block:True,other:True,'self_info':personal,'out_of_material':missing}
                return next((label for name,label in _priority(seek) if flags.get(name)),Label.NORMAL)
            if irrelevant:self.assertEqual(chosen(True),chosen(False))
            else:self.assertNotEqual(chosen(True),chosen(False))
        self.assertFalse(material_irrelevant([{'category':'jailbreak_injection'}]*2,[{'category':'none'}],3))
        self.assertFalse(material_irrelevant([{}],[{'category':'none','self_info':True}],1))
        self.assertFalse(material_irrelevant([{'category':'invalid'}],[{}],1))

    async def test_actual_voted_classification_cancels_only_irrelevant_material(self):
        p=self.pipeline(CancelGuardMaterial);s=self.state(p,True);cancelled=asyncio.Event()
        p._vote_group=AsyncMock(side_effect=lambda state,name,*args: [{'category':'jailbreak_injection' if name=='block' else 'none','self_info':False}]*2)
        async def material(*args):
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        p._ask=material
        await asyncio.wait_for(Voted._classify(p,s),1)
        self.assertIs(s.label,Label.JAILBREAK_INJECTION)
        self.assertTrue(cancelled.is_set())
        self.assertTrue(s._r7_material_cancelled)
        self.assertEqual(s.errors,[])

    async def test_self_info_waits_for_material_and_completed_results_are_preserved(self):
        for category,quote in [('none',''),('none','근거'),('jailbreak_injection','근거')]:
            p=self.pipeline(CancelGuardMaterial);s=self.state(p,True)
            p._vote_group=AsyncMock(side_effect=lambda state,name,*args: [{'category':category if name=='block' else 'none','self_info':True}]*2)
            async def material(*args):return {'quote':quote,'section':'A'}
            p._ask=material
            blocks,others,result=await p._collect_checks(s,'check','material')
            self.assertEqual(result,{'quote':quote,'section':'A'})
            self.assertFalse(getattr(s,'_r7_material_cancelled',False))
            p._collect_checks=AsyncMock(return_value=(blocks,others,result))
            await Voted._classify(p,s)
            self.assertIs(s.label,Label.JAILBREAK_INJECTION if category!='none' else
                          Label.OUT_OF_MATERIAL if not quote else Label.PII_DISCLOSURE)

    async def test_external_cancellation_cleans_all_child_calls(self):
        p=self.pipeline(CancelGuardMaterial);s=self.state(p,True)
        started=[];cancelled=[];ready=asyncio.Event()
        async def wait(state,name,*args):
            started.append(name)
            if len(started)==3:ready.set()
            try:await asyncio.Event().wait()
            finally:cancelled.append(name)
        p._vote_group=p._ask=wait
        task=asyncio.create_task(p._collect_checks(s,'check','material'))
        await asyncio.wait_for(ready.wait(),1);task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertCountEqual(cancelled,['block','other','material'])
        self.assertFalse(getattr(s,'_r7_material_cancelled',False))

    async def test_scope_and_endpoint_marker_are_explicit(self):
        for cls,scope in [(CancelGuardMaterial,'guard'),(CancelTutorMaterial,'tutor'),(CancelBothMaterial,'both')]:
            for guard in (True,False):
                p=self.pipeline(cls);s=self.state(p,guard)
                expected=scope=='both' or guard==(scope=='guard')
                self.assertEqual(p._cancel_target(s),expected)
                if not expected:
                    with patch.object(CharMLP50,'_collect_checks',return_value=('original',)) as base:
                        self.assertEqual(await p._collect_checks(s,'check','material'),('original',))
                        base.assert_awaited_once_with(s,'check','material')
                del s._material_budget_guard
                self.assertFalse(p._cancel_target(s))
            for method in ('_search','_generate','guard','_fusion_safety','_fusion_ground'):
                self.assertIs(getattr(cls,method),getattr(CharMLP50,method))


if __name__=='__main__':unittest.main()
