import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock,patch
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round7_binary_guard import BinaryGuard
from pipelines.kimjaehak.round8_output_veto import OutputVeto


class VetoTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_remote_unsafe_and_uses_no_extra_chat(self):
        p=OutputVeto.__new__(OutputVeto);p.output_risk=Mock();p.output_risk_index=1
        for safe,prob,expected,called in [(True,.95,False,True),(True,.85,True,True),(False,.01,False,False)]:
            p.output_risk.reset_mock();p.output_risk.probabilities.return_value=[[1-prob,prob]]
            s=SimpleNamespace(safe=safe,candidate_answer='candidate',errors=[]);i=SimpleNamespace()
            with patch.object(BinaryGuard,'_run_safety',new_callable=AsyncMock) as base:
                await p._run_safety(s,i,None,0)
                base.assert_awaited_once_with(p,s,i,None,0)
            self.assertEqual(s.safe,expected)
            self.assertEqual(p.output_risk.probabilities.called,called)
            self.assertEqual(getattr(i,'_r8_output_veto',False),safe and not expected)

    async def test_invalid_scores_preserve_B_verdict(self):
        p=OutputVeto.__new__(OutputVeto);p.output_risk=Mock();p.output_risk_index=1
        for scores in [[float('nan'),1],[1],[-.1,1.1],[.6,.6]]:
            p.output_risk.probabilities.return_value=[scores]
            s=SimpleNamespace(safe=True,candidate_answer='candidate',errors=[])
            with patch.object(BinaryGuard,'_run_safety',new_callable=AsyncMock):
                await p._run_safety(s,SimpleNamespace(),None,0)
            self.assertTrue(s.safe);self.assertEqual(s.errors,['output_veto:ValueError'])

    def test_scope(self):
        self.assertIs(load('jh_r8_output_veto'),OutputVeto)
        self.assertIs(OutputVeto._classify,BinaryGuard._classify)
        self.assertIs(OutputVeto.tutor,BinaryGuard.tutor)
        self.assertEqual(OutputVeto.guard_deadline,28)
        self.assertFalse(OutputVeto.guard_safety_launch_early)


if __name__=='__main__':unittest.main()
