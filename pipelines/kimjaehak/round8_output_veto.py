"""High-confidence public-trained output veto; preserve every B unsafe verdict.

Unlike RiskSafety, this adds no independent GPT call. The score is a learned
operating point, not calibrated confidence. Existing A/B remain unchanged.
"""
import numpy as np
from .round8_risk_safety import RiskSafety
from .round7_binary_guard import BinaryGuard


class OutputVeto(RiskSafety):
    guard_safety_launch_early=False
    guard_output_veto_threshold=.9

    async def guard(self,body):
        state=await super().guard(body)
        state.debug['r8_output_veto']=getattr(state,'_r8_output_veto',False)
        return state

    async def _run_safety(self,safety_state,input_state,input_task,started):
        await BinaryGuard._run_safety(self,safety_state,input_state,input_task,started)
        if not safety_state.safe:return
        try:
            p=np.asarray(self.output_risk.probabilities([safety_state.candidate_answer])[0],dtype=float)
            if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
                raise ValueError('Invalid output veto distribution')
            if p[self.output_risk_index]>=self.guard_output_veto_threshold:
                safety_state.safe=False
                input_state._r8_output_veto=True
        except Exception as exc:
            safety_state.errors.append('output_veto:'+type(exc).__name__)

