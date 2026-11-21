"""Restrict optional numeric explanation to quantities, excluding grade numbers."""
import re
from .round8_numeric import BinaryValuePrompt

QUANTITY=re.compile(r'\d[\d,]*(?:\.\d+)?\s*(?:억|만|천|백)?\s*(?:원|%|퍼센트|개월|년|회|번|일|점|포인트|권|개)')


class QuantityValuePrompt(BinaryValuePrompt):
    tutor_apply_quantity_only=True

    def _numeric_request(self,state):
        return super()._numeric_request(state) and bool(QUANTITY.search(state.last_user))
