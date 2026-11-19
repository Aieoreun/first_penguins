"""Narrow tutor check for requested tax values explicitly excluded by the KB.

No amounts, product names, section IDs or question/answer lookup are stored.
This is a conservative language rule, not general semantic verification.
"""
import re
from core.schema import Label
from .classifier_matrix import CharMLP50
from .plain_abstain import ABSTAIN
from .round7_material_scope import ScopedConcrete

ACTUAL=re.compile(r'세후|세율|after[ -]?tax|세금[^.!?\n]{0,20}(?:뺀|공제|차감|반영한|적용한)',re.I)
AFTER=re.compile(r'세후|after[ -]?tax|세금[^.!?\n]{0,20}(?:뺀|공제|차감|반영한|적용한)',re.I)
EXCLUDE=re.compile(r'세금(?:은|을|는|도)?\s*(?:(?:계산|고려)(?:을|를)?\s*하지\s*않|(?:계산에서\s*)?제외)',re.I)
RATE=re.compile(r'(?:세율|소득세율|원천징수율|세금)[^.!?\n]{0,16}\d+(?:\.\d+)?\s*(?:%|퍼센트)',re.I)
PRETAX=re.compile(r'세전|세금(?:은|을)?\s*(?:없이|무시|고려하지|제외하고|빼기\s*전)',re.I)


def excluded_tax_request(question,material):
    if not isinstance(question,str) or not isinstance(material,str):return False
    if not re.search(r'\d',question) or not ACTUAL.search(question):return False
    if PRETAX.search(question) and not AFTER.search(question):return False
    # Provided assumptions or an explicit rate leave interpretation to the
    # original pipeline. Absence of a regex match never establishes support.
    if RATE.search(question) or RATE.search(material):return False
    return bool(EXCLUDE.search(material))


def apply_scope(pipeline,state):
    if (getattr(state,'_material_budget_guard',True) is not False
            or state.label not in {Label.NORMAL,Label.PII_DISCLOSURE}
            or not state.section
            or not state.answer.startswith('관련 교재 내용을 인용할게요.\n\n')):
        return False
    # A rate supplied earlier in the conversation is also a valid assumption.
    if any(RATE.search(m.get('content','')) for m in state.messages):
        return False
    if not excluded_tax_request(state.last_user,pipeline.searcher.section_text(state.section)):
        return False
    state.label=Label.OUT_OF_MATERIAL
    state.answer,state.citations=ABSTAIN,[]
    state.debug['r7_explicit_tax_scope']=True
    return True


class ExplicitTaxScope(CharMLP50):
    tutor_explicit_tax_scope=True

    async def _generate(self,state):
        await super()._generate(state)
        apply_scope(self,state)


class TaxScopedConcrete(ScopedConcrete):
    tutor_explicit_tax_scope=True

    async def _explain(self,state):
        if apply_scope(self,state):return
        await super()._explain(state)


class TaxBudget12(TaxScopedConcrete):
    # NumericApply still clamps this to the original 26-second request deadline.
    tutor_apply_timeout=12.0
