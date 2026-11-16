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
                              "jh_brief_material", "jh_concise_answer", "jh_reasoned_tie", "jh_learned_gate",
                              "jh_learned_gate98", "jh_learned_gate95", "jh_learned_sections", "jh_learned_sections_all",
                              "jh_learned_sections_tutor", "jh_learned_sections_tutor95",
                              "jh_kb_expand", "jh_kb_expand_fitted", "jh_concise_fitted", "jh_evidence_answer",
                              "jh_learned_harmful", "jh_context_output", "jh_context_output3", "jh_plain_abstain",
                              "jh_stronger95", "jh_stronger98", "jh_candidate", "jh_candidate_stronger", "jh_brief_policy",
                              "jh_medium_answer", "jh_clear_policy", "jh_voted_output", "jh_uncertain_sources",
                              "jh_context_policy", "jh_context_policy_voted", "jh_voted_output5",
                              "jh_resilient_sources", "jh_six_sources", "jh_learned_output", "jh_learned_output5",
                              "jh_learned_output_stronger", "jh_guard_category", "jh_context_grounding",
                              "jh_context_grounding_medium", "jh_hybrid_output",
                              "jh_refusal_review", "jh_hybrid_refusal", "jh_hybrid_refusal_broad",
                              "jh_guard_fitted_hybrid", "jh_guard_fitted_fast", "jh_guard_fitted_contextfast",
                              "jh_guard_fitted_span99", "jh_r2_sources", "jh_r2_rrf", "jh_r2_rrf_sources",
                              "jh_r2_examples", "jh_r2_rrf_examples", "jh_r2_input_examples", "jh_r2_dense_input",
                              "jh_r2_candidate", "jh_r2_gate98", "jh_r2_gate995", "jh_r2_joint_input",
                              "jh_r2_joint_gate98", "jh_r2_joint_ungated", "jh_r2_joint_block",
                              "jh_r2_joint_block_ungated", "jh_r2_source_choice", "jh_r2_policy_review",
                              "jh_r2_policy_explained", "jh_r2_policy_complied",
                              "jh_r2_ground2", "jh_r2_ground3", "jh_r2_dense_ungated", "jh_r2_dense98", "jh_r2_dense10",
                              "jh_r2_chat128", "jh_r2_tutor_split", "jh_r2_answer_review",
                              "jh_r2_tutor_dense", "jh_r2_tutor_material10", "jh_r2_tutor_block",
                              "jh_r2_factual_scaffold", "jh_r2_english_answer",
                              "jh_r3_answer_choice", "jh_r3_answer_pair", "jh_r3_tutor_ranked",
                              "jh_r3_quote_answer", "jh_r3_complete_answer", "jh_r3_resolved_query",
                              "jh_r3_answer_longest", "jh_r3_query_gate", "jh_r3_resolved_minimal",
                              "jh_r3_checked_answer", "jh_r3_sentence_answer", "jh_r3_topic_select",
                              "jh_r3_section90", "jh_r3_dual_query", "jh_r3_tutor_fit3000")


def configure():
    # Called before importing v1/v2: their controls are module-level constants.
    os.environ.update(BASELINE_ENV)
