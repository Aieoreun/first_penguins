import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from pipelines.kimjaehak.parallel_guard import Pipeline
from pipelines.kimjaehak.material_budget import Pipeline as MaterialBudget

class VoteFirstTests(unittest.IsolatedAsyncioTestCase):
 async def test_guard_material_starts_after_both_vote_groups(self):
  p=Pipeline.__new__(Pipeline);done=set();both=asyncio.Event()
  async def votes(state,name,system,check):
   done.add(name)
   if len(done)==2:both.set()
   await asyncio.wait_for(both.wait(),.3)
   return [name]
  async def material(*args):
   self.assertEqual(done,{'block','other'})
   return {'quote':'source'}
  p._vote_group=AsyncMock(side_effect=votes);p._ask=AsyncMock(side_effect=material)
  out=await p._collect_checks(SimpleNamespace(_material_budget_guard=True),'check','material')
  self.assertEqual(out,(['block'],['other'],{'quote':'source'}))
  p._ask.assert_awaited_once()
  self.assertEqual(p._ask.await_args.args[-1], 13.0)
 async def test_tutor_delegates_unchanged_collect_checks(self):
  p=Pipeline.__new__(Pipeline);expected=(['b'],['o'],{'quote':'q'})
  with patch.object(MaterialBudget,'_collect_checks',new_callable=AsyncMock,return_value=expected) as original:
   result=await p._collect_checks(SimpleNamespace(_material_budget_guard=False),'c','m')
  self.assertEqual(result,expected);original.assert_awaited_once()
 def test_guard_budget_and_tutor_unchanged(self):
  self.assertEqual(Pipeline.guard_prepare_limit,26)
  self.assertEqual(Pipeline.guard_deadline,28)
  self.assertEqual(Pipeline.guard_material_timeout,13)
  self.assertEqual(Pipeline.deadline,26)
  self.assertIs(Pipeline.tutor,MaterialBudget.tutor)
  self.assertEqual(Pipeline.prepare_limit,MaterialBudget.prepare_limit)
