"""Use public-output risk features to route an independent safety check.

The learned score is only a call-routing signal. It never directly overrides a
safe/unsafe verdict. Low-risk candidates retain B's existing routed safety path.
"""
from copy import copy
import time
import numpy as np
from .classifier_models import CategoryModel
from .round7_binary_guard import BinaryGuard
from .span_tagger import public_rows
from .peace_guard_fusion import Pipeline as Fusion
from .peace_guard_prompt import SAFETY_SYSTEM


class RiskSafety(BinaryGuard):
    guard_safety_launch_early=True
    guard_output_risk_threshold=.1
    guard_output_risk_c=100

    def __init__(self,searcher,llm):
        super().__init__(searcher,llm)
        rows=[r for r in public_rows() if isinstance(r.get('candidate_answer'),str) and r['candidate_answer'].strip()]
        self.output_risk=CategoryModel('char_lr',c=self.guard_output_risk_c).fit(
            [r['candidate_answer'] for r in rows],['safe' if r['labels']['safe'] else 'unsafe' for r in rows])
        self.output_risk_index=int(np.flatnonzero(self.output_risk.classes=='unsafe')[0])

    async def guard(self,body):
        state=await super().guard(body)
        state.debug['r8_risk_safety']=getattr(state,'_r8_risk_safety',False)
        return state

    async def _run_safety(self,safety_state,input_state,input_task,started):
        try:
            p=np.asarray(self.output_risk.probabilities([safety_state.candidate_answer])[0],dtype=float)
            if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
                raise ValueError('Invalid safety routing probability')
            risk=float(p[self.output_risk_index])
        except Exception as exc:
            safety_state.errors.append('risk_router:'+type(exc).__name__)
            # Router failure requests a real check, not an unchecked safe verdict.
            risk=1.
        if risk<self.guard_output_risk_threshold:
            return await super()._run_safety(safety_state,input_state,input_task,started)
        remaining=self.guard_deadline-(time.monotonic()-started)
        if remaining<=0:
            safety_state.safe=False;safety_state.errors.append('safety:NoBudget');return
        safety_state.debug['r8_risk_safety']=True
        input_state._r8_risk_safety=True
        bounded=copy(self);bounded.guard_safety_system=SAFETY_SYSTEM
        bounded.guard_safety_timeout=min(13.,remaining)
        await self._fusion_stage(Fusion._fusion_safety(bounded,safety_state),
                                bounded.guard_safety_timeout,safety_state,'safety')


class RiskSafety50(RiskSafety):
    guard_output_risk_threshold=.5
