import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,AsyncMock,patch
from core.schema import Label
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round8_risk_safety import RiskSafety,RiskSafety50,CompactRiskSafety,COMPACT_SAFETY_SYSTEM
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.peace_guard_prompt import SAFETY_SYSTEM
from tests import test_jh_neural_peace_guard as fixtures

class RiskTests(unittest.IsolatedAsyncioTestCase):
    def ready(self,prob=.9):
        p=fixtures.GuardTests().pipeline(RiskSafety);p.output_risk=SimpleNamespace(probabilities=Mock(return_value=[[1-prob,prob]]));p.output_risk_index=1
        return p

    async def test_high_risk_checks_original_candidate_before_input(self):
        p=self.ready();started=asyncio.Event();body=fixtures.GuardTests().body()
        async def classify(s):await started.wait();s.label=Label.NORMAL
        async def chat(system,user,**kw):
            if kw['max_tokens']==1800:
                self.assertEqual(system,SAFETY_SYSTEM);self.assertEqual(json.loads(user)['candidate_answer'],body['candidate_answer']);started.set();return '{"safe":false}'
            return '{"unsupported":[]}'
        p._classify=classify;p.llm.chat.side_effect=chat
        state=await asyncio.wait_for(p.guard(body),1)
        self.assertFalse(state.safe);self.assertEqual(state.label,Label.NORMAL)

    async def test_low_risk_uses_original_route(self):
        p=self.ready(.01);state=p._new_state(fixtures.GuardTests().body(),True)
        with patch.object(BinaryGuard,'_run_safety',new_callable=AsyncMock) as original:
            await p._run_safety(state,state,None,time.monotonic());original.assert_awaited_once()
        p.llm.chat.assert_not_awaited()

    async def test_router_failure_does_not_claim_safe(self):
        p=self.ready();p.output_risk.probabilities.side_effect=ValueError('bad')
        p.llm.chat.return_value='{"safe":false}'
        state=p._new_state(fixtures.GuardTests().body(),True)
        await p._run_safety(state,state,None,time.monotonic())
        self.assertFalse(state.safe);self.assertIn('risk_router:ValueError',state.errors)
        self.assertEqual(p.llm.chat.await_count,1)

    async def test_compact_prompt_keeps_original_payload_and_budget(self):
        p=fixtures.GuardTests().pipeline(CompactRiskSafety)
        p.output_risk=SimpleNamespace(probabilities=Mock(return_value=[[.1,.9]]));p.output_risk_index=1
        p.llm.chat.return_value='{"safe":true}'
        state=p._new_state(fixtures.GuardTests().body(),True)
        await p._run_safety(state,state,None,time.monotonic())
        call=p.llm.chat.call_args
        self.assertEqual(call.args[0],COMPACT_SAFETY_SYSTEM)
        self.assertEqual(json.loads(call.args[1]),{'conversation':state.messages,'candidate_answer':state.candidate_answer})
        self.assertLessEqual(call.kwargs['timeout'],13)
        self.assertEqual(RiskSafety.risk_safety_system,SAFETY_SYSTEM)

    def test_tutor_input_grounding_unchanged(self):
        for name,cls in [('jh_r8_risk10',RiskSafety),('jh_r8_risk50',RiskSafety50),('jh_r8_risk_compact',CompactRiskSafety)]:
            self.assertIs(load(name),cls)
            for method in ['tutor','_generate','_classify','_fusion_ground','_search']:
                self.assertIs(getattr(cls,method),getattr(BinaryGuard,method))
            self.assertEqual(cls.guard_deadline,28)

if __name__=='__main__':unittest.main()
