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
INITIAL_VARIANTS = ("jh_baseline", "jh_peace_input", "jh_peace_output", "jh_peace_both")
VARIANTS = INITIAL_VARIANTS + ("jh_guided_input", "jh_guided_output", "jh_guided_both", "jh_guided_focus",
                              "jh_v4_output", "jh_precise_output", "jh_calibrated", "jh_v4_safety", "jh_voted",
                              "jh_fewshot", "jh_reasoned", "jh_compact", "jh_explicit", "jh_intent",
                              "jh_diverse", "jh_multiground", "jh_material_review", "jh_block5", "jh_reconsider",
                              "jh_unanimous", "jh_token_output", "jh_adaptive", "jh_adaptive_unanimous",
                              "jh_medium_output", "jh_thin_material", "jh_adaptive_thin", "jh_adaptive_explicit",
                              "jh_brief_material")


def configure():
    # Called before importing v1/v2: their controls are module-level constants.
    os.environ.update(BASELINE_ENV)
