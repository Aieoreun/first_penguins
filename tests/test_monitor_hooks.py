import asyncio,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
sys.path.insert(0,'/Users/peace/Downloads/first_penguins-main');sys.path.insert(0,str(Path(__file__).resolve().parent))
import guard_supervisor as gs
from guard_review import OutputReviewReport
from guard_output import GroundingVerdict
C='값은 1입니다.'
E=[{'section_id':'x','text':'값은 4입니다.'}]
I={'input':{'decision':'allow','category':'normal'},'output':{'grounded':True,'unsupported':[],'safe':True,'category':'normal'}}
R={'grounding':[{'candidate_span':C,'relation':'contradicted','reason':'불일치','citations':[{'source':'kb','source_id':'x','quote':E[0]['text']}]}],'safety':{'candidate_span':C,'relation':'benign','reason':'설명'}}
class MonitorTests(unittest.IsolatedAsyncioTestCase):
 async def run_case(self,observer):
  s=gs.GuardSupervisor(None,'m',SimpleNamespace(search=AsyncMock()),observer=observer)
  with patch.object(gs,'review_verdict',AsyncMock(return_value=OutputReviewReport.model_validate(R))),patch.object(gs,'check_grounding',AsyncMock(return_value=GroundingVerdict(grounded=False,unsupported=[C]))):
   return await s._review_output([],C,E,I,time.monotonic()+5)
 async def test_trace_does_not_change_decision_and_has_action(self):
  events=[]
  result=await self.run_case(lambda event,**data:events.append({'event':event,**data}))
  self.assertEqual(result,await self.run_case(None))
  self.assertIn('recheck',[e.get('action') for e in events]);self.assertIn('apply',[e.get('action') for e in events])
 async def test_broken_observer_does_not_change_verdict(self):
  def fail(*a,**kw):raise ValueError('monitor failure')
  self.assertEqual(await self.run_case(fail),await self.run_case(None))
if __name__=='__main__':unittest.main()
