"""Guard-only local fallbacks when every relevant remote check failed.

Successful or partially successful remote decisions are never replaced. This
is an opt-in failure-mode experiment, not a lower threshold during healthy use.
"""
import numpy as np
from core.schema import Decision,Label,parse_label
from .round7_binary_guard import BinaryGuard
from .round7_binary_model import BLOCKED


def input_checks_failed(state):
    return (getattr(state,'_material_budget_guard',False) is True
        and state.label is Label.NORMAL
        and state.debug.get('valid_block_votes')==0
        and state.debug.get('valid_other_votes')==0
        and not state.debug.get('self_info_votes')
        and not state.debug.get('out_of_material')
        and all(any(e.startswith(name+':') for e in state.errors) for name in ('block','other','material')))


class FailureInput(BinaryGuard):
    guard_failure_input_threshold=.5

    async def _classify(self,state):
        await super()._classify(state)
        if not input_checks_failed(state):return
        try:
            p=np.asarray(self.matrix_model.probabilities([state.last_user])[0],dtype=float)
            if p.shape!=(len(self.matrix_model.classes),) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
                raise ValueError('Invalid local fallback distribution')
            if p.max()<self.guard_failure_input_threshold:return
            category=str(self.matrix_model.classes[p.argmax()])
            # Both harmful subtypes have identical guard category/decision.
            label=Label.HARMFUL_REQUEST if category=='harmful_content' else parse_label(category)
            if label is None:return
            state.label=label;state.debug['r8_failure_input']=True
        except Exception as exc:
            state.errors.append('failure_input:'+type(exc).__name__)


class FailureGround(BinaryGuard):
    guard_failure_span_threshold=.9
    guard_failure_span_max_chars=4000

    async def guard(self,body):
        state=await super().guard(body)
        state.debug['r8_failure_ground']=(state.debug.get('fusion_completed_ground_votes')==0 and state.debug.get('learned_span') is True)
        return state

    async def _fusion_ground(self,state):
        await super()._fusion_ground(state)
        if (state.debug.get('fusion_completed_ground_votes')!=0 or not state.errors
                or not state.candidate_answer or len(state.candidate_answer)>self.guard_failure_span_max_chars):return
        try:
            spans,confidence=self.span_tagger.predict(state.candidate_answer)
            if not np.isfinite(confidence) or confidence<self.guard_failure_span_threshold:return
            if not isinstance(spans,list) or any(not isinstance(s,str) or not s or s not in state.candidate_answer for s in spans):
                raise ValueError('Invalid fallback evidence spans')
            state.unsupported=spans;state.grounded=not spans;state.debug['r8_failure_ground']=True;state.debug['learned_span']=True
        except Exception as exc:
            state.errors.append('failure_ground:'+type(exc).__name__)


class FailureBoth(FailureInput,FailureGround):
    pass


class FailureBlock(BinaryGuard):
    """Use local blocked mass only when the entire remote BLOCK ballot failed.

Valid remote BLOCK ballots (including none) and existing BLOCK decisions win.
Unlike FailureInput, this may supersede an OTHER/material ALLOW label: BLOCK
categories have priority, but only when the missing BLOCK check is evidenced.
"""
    guard_failure_block_threshold=.7

    async def _classify(self,state):
        await super()._classify(state)
        if (getattr(state,'_material_budget_guard',False) is not True
                or state.decision is Decision.BLOCK
                or state.debug.get('valid_block_votes')!=0
                or not any(e.startswith('block:') for e in state.errors)):
            return
        try:
            p=np.asarray(self.matrix_model.probabilities([state.last_user])[0],dtype=float)
            classes=self.matrix_model.classes
            if (p.shape!=(len(classes),) or not np.isfinite(p).all()
                    or np.any(p<0) or not np.isclose(p.sum(),1)):
                raise ValueError('Invalid local block distribution')
            indexes=[i for i,c in enumerate(classes) if c in BLOCKED]
            if not indexes or sum(p[i] for i in indexes)<self.guard_failure_block_threshold:return
            category=str(classes[max(indexes,key=lambda i:p[i])])
            label=Label.HARMFUL_REQUEST if category=='harmful_content' else parse_label(category)
            if label is None:return
            state.label=label;state.debug['r8_failure_block']=True
        except Exception as exc:
            state.errors.append('failure_block:'+type(exc).__name__)
