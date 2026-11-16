import asyncio
import copy
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError
import guard_review as gr
import guard_supervisor as gs
import guard_output as go

C='값은 1입니다.'
E=[{'section_id':'unit','text':'값은 4입니다.'}]
M=[{'role':'user','content':'값은 얼마인가요?'}]

def initial(grounded=False,safe=True):
 return {'input':{'decision':'allow','category':'normal'},'output':{'grounded':grounded,'unsupported':[] if grounded else [C],'safe':safe,'category':'normal'}}

def observation(relation='contradicted',safety='benign',span=C):
 cites=[]
 if relation in ('supported','contradicted'):cites.append({'source':'kb','source_id':'unit','quote':E[0]['text']})
 return gr.OutputReviewReport.model_validate({'grounding':[{'candidate_span':span,'relation':relation,'reason':'관계 대조','citations':cites}], 'safety':{'candidate_span':C,'relation':safety,'reason':'답변 행동'}})

class SchemaTests(unittest.TestCase):
 def test_final_values_and_actions_forbidden(self):
  data=observation().model_dump()
  for k,v in [('grounded',True),('safe',True),('action','revise'),('unsupported',[])]:
   bad=copy.deepcopy(data);bad['grounding'][0][k]=v
   with self.assertRaises(ValidationError):gr.OutputReviewReport.model_validate(bad)
 def test_schema_cannot_return_verdict(self):
  s=gr.review_json_schema(targets=('grounding','safety'))
  def walk(n):
   if isinstance(n,dict):
    props=n.get('properties',{})
    self.assertFalse(set(props)&{'grounded','safe','unsupported','action','input'})
    for v in n.values():walk(v)
   elif isinstance(n,list):
    for v in n:walk(v)
  walk(s)

class SupervisorTests(unittest.IsolatedAsyncioTestCase):
 async def asyncSetUp(self):
  self.search=SimpleNamespace(search=AsyncMock(return_value=E))
  self.s=gs.GuardSupervisor(None,'model',self.search)
  self.r=patch.object(gs,'review_verdict',new_callable=AsyncMock); self.review=self.r.start()
  self.g=patch.object(gs,'check_grounding',new_callable=AsyncMock);self.ground=self.g.start()
  self.a=patch.object(gs,'check_safety',new_callable=AsyncMock);self.safe=self.a.start()
  for p in [self.r,self.g,self.a]:self.addCleanup(p.stop)
  self.review.return_value=observation()
 async def run_review(self,ini=None,seconds=20):
  self.ini=ini or initial(); saved=copy.deepcopy(self.ini)
  result=await self.s._review_output(M,C,E,self.ini,time.monotonic()+seconds)
  self.assertEqual(saved,self.ini)
  return result
 async def test_contradiction_agrees_keep_without_rerun(self):
  out=await self.run_review()
  self.assertFalse(out['grounded']);self.ground.assert_not_awaited();self.safe.assert_not_awaited()
 async def test_conflict_rechecks_only_grounding_with_feedback(self):
  self.ground.return_value=go.GroundingVerdict(grounded=False,unsupported=[C])
  out=await self.run_review(initial(True))
  self.assertFalse(out['grounded']);self.ground.assert_awaited_once();self.safe.assert_not_awaited()
  self.assertIn('observations',self.ground.call_args.kwargs['review_feedback'])
 async def test_recheck_disagreement_preserves_initial(self):
  self.ground.return_value=go.GroundingVerdict(grounded=True,unsupported=[])
  out=await self.run_review(initial(True));self.assertTrue(out['grounded']);self.ground.assert_awaited_once()
 async def test_nonfactual_recheck_can_clear(self):
  self.review.return_value=observation('nonfactual')
  self.ground.return_value=go.GroundingVerdict(grounded=True,unsupported=[])
  self.assertTrue((await self.run_review())['grounded'])
 async def test_partial_clearance_cannot_clear_all(self):
  self.review.return_value=observation('nonfactual',span='값은')
  out=await self.run_review();self.assertFalse(out['grounded']);self.ground.assert_not_awaited()
 async def test_missing_quote_blocks_recheck(self):
  self.review.return_value.grounding[0].citations[-1].quote='없는 자료'
  out=await self.run_review(initial(True));self.assertTrue(out['grounded']);self.ground.assert_not_awaited()
 async def test_safety_conflict_rechecks_only_safety(self):
  self.review.return_value=observation(safety='harmful')
  self.safe.return_value=go.SafetyVerdict(safe=False)
  out=await self.run_review();self.assertFalse(out['safe']);self.safe.assert_awaited_once();self.ground.assert_not_awaited()
 async def test_insufficient_no_new_evidence_keeps(self):
  self.review.return_value=observation('insufficient')
  await self.run_review();self.search.search.assert_awaited_once();self.ground.assert_not_awaited();self.review.assert_awaited_once()
 async def test_insufficient_new_evidence_rechecks_once(self):
  self.review.return_value=observation('insufficient')
  self.search.search.return_value=[{'section_id':'extra','text':'추가 자료'}]
  self.ground.return_value=go.GroundingVerdict(grounded=True,unsupported=[])
  self.assertTrue((await self.run_review())['grounded']);self.ground.assert_awaited_once();self.review.assert_awaited_once()
 async def test_recheck_failure_does_not_retry(self):
  self.ground.side_effect=ValueError('bad')
  self.assertTrue((await self.run_review(initial(True)))['grounded']);self.ground.assert_awaited_once()
 async def test_timeout_preserves(self):
  self.s.review_timeout=.01
  async def slow(*a,**kw):await asyncio.sleep(1)
  self.ground.side_effect=slow
  self.assertTrue((await self.run_review(initial(True)))['grounded']);self.ground.assert_awaited_once()
 async def test_one_failure_does_not_undo_other_target(self):
  self.review.return_value=observation(safety='harmful')
  self.ground.side_effect=ValueError('bad');self.safe.return_value=go.SafetyVerdict(safe=False)
  out=await self.run_review(initial(True));self.assertTrue(out['grounded']);self.assertFalse(out['safe'])
 async def test_legacy_revision_not_accepted(self):
  self.review.return_value=gr.ReviewReport.model_validate({'reviews':[{'target':'grounding','action':'revise','rule':'factual_support','reason':'bad','citations':[],'grounding':{'grounded':True,'unsupported':[]}}]})
  self.assertFalse((await self.run_review())['grounded']);self.ground.assert_not_awaited()
 async def test_bad_span_not_accepted_from_recheck(self):
  self.ground.return_value=go.GroundingVerdict(grounded=False,unsupported=['없는 문장'])
  self.assertTrue((await self.run_review(initial(True)))['grounded'])

 async def test_benign_partial_span_cannot_clear_safety(self):
  self.review.return_value=observation()
  self.review.return_value.safety.candidate_span='값은'
  out=await self.run_review(initial(False,False))
  self.assertFalse(out['safe']);self.safe.assert_not_awaited()
 async def test_rechecks_are_parallel_and_cancelled_with_parent(self):
  self.review.return_value=observation(safety='harmful')
  entered=set();cancelled=set();ready=asyncio.Event()
  def slow(name):
   async def call(*a,**kw):
    entered.add(name)
    if len(entered)==2:ready.set()
    try:await asyncio.sleep(5)
    finally:cancelled.add(name)
   return call
  self.ground.side_effect=slow('grounding');self.safe.side_effect=slow('safety')
  task=asyncio.create_task(self.run_review(initial(True)))
  await asyncio.wait_for(ready.wait(),.5);task.cancel()
  with self.assertRaises(asyncio.CancelledError):await task
  self.assertEqual(cancelled,{'grounding','safety'})
 async def test_deadline_skips_all_calls(self):
  out=await self.run_review(seconds=-1)
  self.assertEqual(out,initial()['output']);self.review.assert_not_awaited()

class ModelCallTests(unittest.IsolatedAsyncioTestCase):
 async def test_first_pass_payload_unchanged_recheck_is_data(self):
  create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"unsupported_ids":[]}'))]))
  client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
  await go.check_grounding(client,'model',M,C,E)
  base=create.call_args.kwargs
  self.assertEqual(base['messages'][0]['content'],go.GROUNDING_POLICY)
  self.assertNotIn('review_feedback',json.loads(base['messages'][1]['content']))
  await go.check_grounding(client,'model',M,C,E,review_feedback={'observations':[]})
  feedback=create.call_args.kwargs
  self.assertIn('review_feedback',json.loads(feedback['messages'][1]['content']))
  self.assertIn('독립적으로',feedback['messages'][0]['content'])
