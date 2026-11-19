"""Combine independently tested changes; keep the original candidates selectable."""
from .concise_answer import Fitted as ConciseAnswer
from .context_output import Pipeline as ContextOutput
from .learned_harmful import Pipeline as HarmfulSubtypes
from .plain_abstain import Pipeline as PlainAbstain


class Pipeline(ContextOutput, PlainAbstain, HarmfulSubtypes, ConciseAnswer):
    # C3 order composes: contextual output, plain abstention, subtype gate,
    # concise answer prompts, then the common tutor-only section classifier.
    pass


class Stronger(Pipeline):
    learned_gate_c = 3000
