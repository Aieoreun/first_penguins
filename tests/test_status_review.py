"""Scoped status-review runtime invariants; no live model or dataset labels."""
import asyncio
import copy
import json
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError
import guard_review as gr
import guard_supervisor as gs
from guard_output import GroundingVerdict, SafetyVerdict

M=[{'role':'user','content':'현재 요청을 설명해 줘'}]
C='후보 문장입니다.'
E=[{'section_id':'section','text':'검증 자료입니다.'}]
I={'input':{'decision':'allow','category':'normal'},'output':{'grounded':True,'unsupported':[],'safe':True,'category':'normal'}}

def ground(status='supported',relation='nonfactual'):
 return gr.GroundingStatusReview(status=status,rule='factual_support',reason='원문 대조',observations=[gr.GroundingObservation(candidate_span=C,relation=relation,reason='관계',citations=[] if relation not in {'supported','contradicted'} else [gr.EvidenceCitation(source='kb',source_id='section',quote=E[0]['text'])])])
def inp(status='supported'):
 return gr.InputStatusReview(status=status,rule='input_intent',reason='요청 의도',citations=[gr.InputCitation(source='conversation',source_id='0',quote=M[0]['content'])])
def safety(status='supported',relation='benign'):
 return gr.SafetyStatusReview(status=status,rule='answer_safety',reason='실제 행동',observation=gr.SafetyObservation(candidate_span=C,relation=relation,reason='행동'))

class Contracts(unittest.IsolatedAsyncioTestCase):
 async def test_payloads_are_scoped_and_no_replacement_fields(self):
  cases=[(gr.review_input_status,[M,I['input']],inp(),{'conversation','initial'}),
         (gr.review_grounding,[M,C,E,{'grounded':True,'unsupported':[]}],ground(),{'conversation','candidate_answer','evidence','initial'}),
         (gr.review_safety,[M,C,{'safe':True}],safety(),{'conversation','candidate_answer','initial'})]
  for fn,args,report,keys in cases:
   create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=report.model_dump_json()))]))
   client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
   await fn(client,'model',*args)
   call=create.call_args.kwargs;payload=json.loads(call['messages'][1]['content'])
   self.assertEqual(set(payload),keys)
   self.assertEqual(payload['initial'],args[-1])
   schema=call['response_format']['json_schema']['schema']
   self.assertEqual(set(schema['properties']['status']['enum']),{'supported','refuted','insufficient'})
   self.assertFalse(set(schema['properties']) & {'safe','grounded','decision','action','input'})
   with self.assertRaises(ValidationError): type(report).model_validate({**report.model_dump(),'safe':False})

class Runtime(unittest.IsolatedAsyncioTestCase):
 def setUp(self):
  self.stack=ExitStack();self.addCleanup(self.stack.close)
  self.m={name:self.stack.enter_context(patch.object(gs,name,new_callable=AsyncMock)) for name in ['review_input_status','review_grounding','review_safety','classify_input','check_grounding','check_safety']}
  self.m['review_input_status'].return_value=inp();self.m['review_grounding'].return_value=ground();self.m['review_safety'].return_value=safety()
  self.m['classify_input'].return_value=gr.InputVerdict(decision='block',category='academic_dishonesty')
  self.m['check_grounding'].return_value=GroundingVerdict(grounded=False,unsupported=[C]);self.m['check_safety'].return_value=SafetyVerdict(safe=False)
  self.search=SimpleNamespace(search=AsyncMock(return_value=E));self.events=[]
  self.s=gs.GuardSupervisor(None,'model',self.search,observer=lambda event,**data:self.events.append({'event':event,**data}))
 async def run_case(self,initial=None,candidate=C):
  return await self.s.review_initial(M,candidate,E,copy.deepcopy(initial or I),time.monotonic()+5)
 async def test_three_calls_parallel_and_frozen(self):
  entered=set();ready=asyncio.Event()
  def call(name,value):
   async def fn(*a,**kw):
    entered.add(name)
    if len(entered)==3:ready.set()
    await asyncio.wait_for(ready.wait(),.3)
    return value
   return fn
  for name in ['review_input_status','review_grounding','review_safety']:self.m[name].side_effect=call(name,self.m[name].return_value)
  initial=copy.deepcopy(I);r=await self.s.review_initial(M,C,E,initial,time.monotonic()+5)
  self.assertEqual(r,I);self.assertEqual(initial,I);self.assertEqual(len(entered),3)
 async def test_refuted_only_rechecks_its_scope_with_status_and_original(self):
  self.m['review_input_status'].return_value=inp('refuted')
  result=await self.run_case();self.assertEqual(result['input']['decision'],'block');self.assertEqual(result['output'],I['output'])
  f=self.m['classify_input'].call_args.kwargs['review_feedback'];self.assertEqual(f['status'],'refuted');self.assertEqual(f['initial'],I['input'])
  self.m['classify_input'].assert_awaited_once();self.m['check_grounding'].assert_not_awaited();self.m['check_safety'].assert_not_awaited()
 async def test_rechecker_can_reject_refutation(self):
  self.m['review_input_status'].return_value=inp('refuted');self.m['classify_input'].return_value=gr.InputVerdict(**I['input'])
  self.assertEqual(await self.run_case(),I)
 async def test_failure_one_scope_does_not_undo_other(self):
  self.m['review_input_status'].side_effect=ValueError('bad');self.m['review_safety'].return_value=safety('refuted','harmful')
  r=await self.run_case();self.assertFalse(r['output']['safe']);self.assertEqual(r['input'],I['input'])
 async def test_insufficient_input_and_safety_keep_record_uncertainty(self):
  self.m['review_input_status'].return_value=inp('insufficient');self.m['review_safety'].return_value=safety('insufficient','uncertain')
  self.assertEqual(await self.run_case(),I);self.m['classify_input'].assert_not_awaited();self.m['check_safety'].assert_not_awaited()
  self.assertEqual(len([e for e in self.events if e.get('reason')=='insufficient_evidence']),2)
 async def test_insufficient_no_new_evidence_no_recheck(self):
  self.m['review_grounding'].return_value=ground('insufficient','insufficient')
  self.assertEqual(await self.run_case(),I);self.search.search.assert_awaited_once();self.m['check_grounding'].assert_not_awaited()
 async def test_insufficient_new_evidence_one_recheck(self):
  self.m['review_grounding'].return_value=ground('insufficient','insufficient');self.search.search.return_value=[{'section_id':'extra','text':'새 자료'}]
  await self.run_case();self.m['check_grounding'].assert_awaited_once()
  self.assertEqual(self.m['check_grounding'].call_args.kwargs['review_feedback']['status'],'insufficient')
 async def test_invalid_citation_never_rechecks(self):
  r=inp('refuted');r.citations[0].quote='없는 인용';self.m['review_input_status'].return_value=r
  self.assertEqual(await self.run_case(),I);self.m['classify_input'].assert_not_awaited()
 async def test_contradiction_prevents_clearing_grounding(self):
  initial=copy.deepcopy(I);initial['output'].update(grounded=False,unsupported=[C])
  r=ground('refuted','nonfactual');r.observations+=ground('refuted','contradicted').observations
  self.m['review_grounding'].return_value=r
  self.assertEqual(await self.run_case(initial),initial);self.m['check_grounding'].assert_not_awaited()
 async def test_no_candidate_only_input(self):
  await self.run_case({'input':I['input']},candidate=None)
  self.m['review_input_status'].assert_awaited_once();self.m['review_grounding'].assert_not_awaited();self.m['review_safety'].assert_not_awaited()
 async def test_cancellation_reaches_all_reviewers(self):
  started=set();cancelled=set();ready=asyncio.Event()
  def slow(name):
   async def fn(*a,**kw):
    started.add(name)
    if len(started)==3:ready.set()
    try:await asyncio.sleep(5)
    finally:cancelled.add(name)
   return fn
  for name in ['review_input_status','review_grounding','review_safety']:self.m[name].side_effect=slow(name)
  task=asyncio.create_task(self.run_case());await asyncio.wait_for(ready.wait(),.3);task.cancel()
  with self.assertRaises(asyncio.CancelledError):await task
  self.assertEqual(cancelled,started)
 async def test_deadline_returns_original_once_no_retry(self):
  self.s.review_timeout=.01
  async def slow(*a,**kw):await asyncio.sleep(5)
  for name in ['review_input_status','review_grounding','review_safety']:self.m[name].side_effect=slow
  self.assertEqual(await self.run_case(),I)
  for name in ['review_input_status','review_grounding','review_safety']:self.m[name].assert_awaited_once()

if __name__=='__main__':unittest.main()
