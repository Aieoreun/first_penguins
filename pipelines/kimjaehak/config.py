"""Pinned v2 controls shared by all four initial experiment variants."""
import os

BASELINE_ENV = {
    "V2_REWRITE": "0", "V2_REWRITE_EFFORT": "minimal", "V2_REWRITE_TIMEOUT": "6",
    "V2_SECTION": "top1", "V2_VOTES": "1", "V2_SEEK_FIRST": "0",
    "V2_MATERIAL_EFFORT": "", "V2_MATERIAL_INPUT": "chunks", "V2_MATERIAL_SECTIONS": "3",
    "V2_PROMPTS": "1", "V2_SAFE_RULE": "0", "V2_ANSWER_EFFORT": "minimal",
    "V2_ANSWER_MAX_TOKENS": "2000", "CLASSIFY_EFFORT": "low",
    "ANSWER_EFFORT": "minimal", "GUARD_EFFORT": "low",
    "EMBED_TIMEOUT": "3", "CLASSIFY_TIMEOUT": "10", "SECOND_TIMEOUT": "13",
}
VARIANTS = ("jh_baseline", "jh_peace_input", "jh_peace_output", "jh_peace_both")


def configure():
    # Called before importing v1/v2: their controls are module-level constants.
    os.environ.update(BASELINE_ENV)
