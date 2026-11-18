"""Isolate the entity/property prompt from hard lexical quote rejection."""
from .round6_joint_evidence import Pipeline as JointEvidence
from .round6_joint_quote import QUOTE_SYSTEM


class Pipeline(JointEvidence):
    tutor_joint_fact = True
    joint_evidence_system = QUOTE_SYSTEM
    # This is the original bool/section adapter. A model-written quote is
    # neither treated as verified evidence nor emitted in the tutor answer.
    # The final answer still uses the actual selected KB section.
