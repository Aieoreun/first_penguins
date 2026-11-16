"""Semantically check every nonempty blocked answer, regardless of refusal words."""
import re

from .guard_fitted import ContextFast


class Pipeline(ContextFast):
    # This triggers the existing semantic policy judge; it is never a safe verdict.
    # Inputs, factual grounding, judge prompt and failure defaults stay inherited.
    all_blocked_policy_review = True
    refusal_pattern = re.compile(r"[\s\S]")
