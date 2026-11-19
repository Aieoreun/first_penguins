"""Run an isolated experiment server using the repository's shared .env."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from experiments.jh_environment import load_environment
    import uvicorn
    from pipelines.kimjaehak.config import VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=(*VARIANTS, "v4"), required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--embedding-mode", choices=("env", "off"), default="env")
    parser.add_argument("--request-slots", type=int, help="Override the candidate's internal request slots (1..64)")
    args = parser.parse_args()
    if args.request_slots is not None and not 1 <= args.request_slots <= 64:
        parser.error("request-slots must be between 1 and 64")
    load_environment(ROOT)
    if args.embedding_mode == "off":
        for key in ("EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_API_KEY"):
            os.environ[key] = ""
    os.environ["PIPELINE"] = args.variant
    for key in ("TEAM_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if not os.environ.get(key):
            raise SystemExit(f"Missing setting: {key}")
    from pipelines import load
    pipeline = load(args.variant)
    dual_metadata = {}
    if getattr(pipeline, "source_dual_granularity", False):
        import asyncio
        from pipelines.kimjaehak.dual_index import prepare_short_index
        dual_metadata = asyncio.run(prepare_short_index())
    if args.request_slots is not None:
        if not hasattr(pipeline, "request_slots"):
            parser.error("This pipeline has no configurable request semaphore")
        pipeline.request_slots = args.request_slots
    fields = ("material_cancel_scope", "tutor_material_scope", "tutor_apply_missing_material_abstain", "guard_adaptive_ground", "tutor_apply_joint_source", "tutor_apply_concrete", "tutor_apply_max_chars", "guard_ground_context", "tutor_apply_numeric", "tutor_apply_all", "tutor_apply_timeout", "classifier_embedding_batched", "classifier_hybrid_dense_weight", "classifier_hybrid_fast_threshold", "classifier_representation", "guard_verdict_first", "r7_product_context", "r7_other_boundary", "r7_scaffold_guide", "reasoning_guard_input_tokens", "reasoning_guard_ground_tokens", "reasoning_guard_input", "reasoning_guard_ground", "reasoning_guard_safety",
              "reasoning_tutor_input", "reasoning_tutor_source", "answer_effort", "answer_max_tokens", "votes", "seek_first", "safe_by_rule",
              "deadline", "prepare_limit", "peace_input", "peace_output",
              "guided_input", "guided_output", "focus_input", "request_slots", "short_votes", "other_vote_count", "block_min_votes",
              "short_effort", "short_max_tokens", "short_timeout", "answer_sections", "material_review_timeout",
              "block_review_timeout", "adaptive_votes", "output_effort", "output_max_tokens", "material_max_chunks",
              "material_max_tokens", "tie_effort", "tie_max_tokens",
              "learned_gate_threshold", "learned_gate_c", "training_source",
              "learned_section_c", "learned_section_threshold", "learned_section_require_retrieved",
              "learned_section_use_for_output", "query_expansion_weight", "evidence_first",
              "learned_harmful_c", "learned_harmful_threshold", "output_sections",
              "stream_answer_effort", "stream_answer_max_tokens", "stream_answer_timeout",
              "fixed_academic_scaffold", "output_votes", "uncertain_sources_only", "contextual_refusal",
              "retry_source_failure", "learned_span_c", "learned_span_threshold", "guard_category_without_subtype",
              "output_timeout", "hybrid_output", "refusal_review", "refusal_timeout", "guard_only_c",
              "span_decision_threshold", "retrieval_fusion", "tutor_example_count",
              "input_example_count", "dense_input_examples", "example_check_names", "joint_input", "joint_input_timeout", "joint_block_only",
              "source_choice", "source_choice_timeout", "source_choice_sections", "all_blocked_policy_review",
              "complied_policy_review", "chat_slots", "answer_review", "review_timeout", "review_min_remaining", "factual_scaffold", "english_answer",
              "answer_choices", "answer_choice_timeout", "answer_choice_min_remaining", "answer_selection",
              "tutor_section_ranker", "tutor_ranker_c", "tutor_ranker_margin",
              "tutor_answer_style", "tutor_resolved_query", "query_resolve_timeout", "query_resolve_effort",
              "tutor_topic_selection", "tutor_query_gate", "tutor_section_threshold",
              "tutor_dual_query", "tutor_answer_support", "tutor_extract_scope", "tutor_material_source", "material_quote_min_chars",
              "source_extraction_selector", "source_extract_timeout", "source_extract_sections", "source_extract_expanded_limit",
              "tutor_other_boundary", "tutor_material_boundary", "source_extract_quote_required", "tutor_material_votes",
              "tutor_safe_alternative", "tutor_safe_menu", "tutor_category_normalization", "source_last_dense", "source_last_dense_timeout", "source_last_dense_sections", "source_conditional_dense", "structured_tutor_block", "source_conversation_last", "tutor_material_timeout", "tutor_prepare_limit", "source_stream_early_finish", "source_stream_buffer_chars", "source_id_only", "source_disambiguation", "source_minimal_effort")
    effective = {key: getattr(pipeline, key, None) for key in fields}
    effective["class"] = pipeline.__module__ + "." + pipeline.__name__
    effective.update(dual_metadata)
    effective["source_dual_granularity"] = getattr(pipeline, "source_dual_granularity", False)
    for key in ("tutor_short_material", "tutor_material_no_section", "tutor_verify_context_sources", "tutor_lexical_mode", "source_evidence_order", "source_selection_votes", "tutor_computable_material", "source_resolved_hint", "source_selection_effort", "source_selection_tokens", "tutor_normal_gate_threshold", "tutor_joint_evidence", "joint_evidence_tokens", "tutor_skip_trusted_embedding", "tutor_joint_quote", "tutor_eager_source", "tutor_joint_fact", "tutor_material_resolved_hint", "tutor_source_support", "source_support_tokens", "source_support_timeout", "tutor_source_conditioned_gate", "source_conditioned_input_threshold", "source_conditioned_section_threshold", "source_conditioned_fallback_threshold", "tutor_product_context"):
        effective[key] = getattr(pipeline, key, None)
    effective["embedding_mode"] = args.embedding_mode
    for key in ("tutor_mlp_threshold", "tutor_mlp_single_turn", "tutor_mlp_fallback",
                "classifier_kind", "classifier_threshold", "classifier_seed", "classifier_width",
                "classifier_alpha", "classifier_c", "classifier_embedding_timeout",
                "classifier_text_mode", "classifier_common_llm_fallback",
                "guard_fusion", "guard_fusion_reference", "guard_independent_safety", "guard_requested_action",
                "guard_safety_action_review", "guard_safety_curriculum_context",
                "guard_safety_after_input",
                "guard_routed_safety", "guard_safety_votes",
                "guard_ground_effort", "guard_ground_max_tokens",
                "guard_fast_input_wait",
                "guard_gate_threshold",
                "guard_cancel_ground_after_block",
                "guard_allow_safety", "guard_allow_safety_timeout",
                "guard_span_review", "guard_span_review_threshold", "guard_span_review_timeout",
                "guard_preserve_completed_ground", "guard_restore_span_refinement",
                "guard_vote_first", "guard_prepare_limit", "guard_deadline",
                "guard_material_timeout", "guard_ground_timeout", "guard_safety_timeout"):
        effective[key] = getattr(pipeline, key, None)
    if hasattr(pipeline, "guard_safety_system"):
        effective["guard_safety_prompt_sha256"] = hashlib.sha256(pipeline.guard_safety_system.encode()).hexdigest()
    effective["guard_safety_hints_sha256"] = {k: hashlib.sha256(v.encode()).hexdigest()
        for k, v in getattr(pipeline, "guard_safety_hint_systems", {}).items()}
    if hasattr(pipeline, "guard_block_system"):
        effective["guard_block_prompt_sha256"] = hashlib.sha256(pipeline.guard_block_system.encode()).hexdigest()
    effective["embedding_model"] = os.environ.get("EMBEDDING_MODEL") or None
    if hasattr(pipeline, "endpoint_variants"):
        effective["endpoint_variants"] = pipeline.endpoint_variants
        effective["shared_request_slots"] = pipeline.request_slots
        effective["tutor_overrides"] = {key: getattr(pipeline.tutor_type, key, None)
                                         for key in fields
                                         if key != "request_slots"
                                         and getattr(pipeline.tutor_type, key, None) != effective[key]}
    if hasattr(pipeline, "refusal_pattern"):
        effective["refusal_pattern"] = pipeline.refusal_pattern.pattern
    effective["answer_prompts_sha256"] = hashlib.sha256(json.dumps(
        {k.value: v for k, v in pipeline.answer_systems.items()}, sort_keys=True).encode()).hexdigest()
    effective["unsupported_prompt_sha256"] = hashlib.sha256(pipeline.unsupported_system.encode()).hexdigest()
    if hasattr(pipeline, "guard_span_review_system"):
        effective["guard_span_review_system_sha256"] = hashlib.sha256(pipeline.guard_span_review_system.encode()).hexdigest()
    if hasattr(pipeline, "policy_system"):
        effective["policy_prompt_sha256"] = hashlib.sha256(pipeline.policy_system.encode()).hexdigest()
    if hasattr(pipeline, "source_choice_system"):
        effective["source_choice_prompt_sha256"] = hashlib.sha256(pipeline.source_choice_system.encode()).hexdigest()
    if hasattr(pipeline, "joint_evidence_system"):
        effective["joint_evidence_prompt_sha256"] = hashlib.sha256(pipeline.joint_evidence_system.encode()).hexdigest()
    if hasattr(pipeline, "source_support_system"):
        effective["source_support_prompt_sha256"] = hashlib.sha256(pipeline.source_support_system.encode()).hexdigest()
    if getattr(pipeline, "source_resolved_hint", False):
        from pipelines.kimjaehak.round6_query_hint import HINT_RULE
        effective["source_resolved_hint_prompt_sha256"] = hashlib.sha256(HINT_RULE.encode()).hexdigest()
    if hasattr(pipeline, "source_extract_system"):
        effective["source_extract_prompt_sha256"] = hashlib.sha256(pipeline.source_extract_system.encode()).hexdigest()
    if hasattr(pipeline, "tutor_block_system"):
        effective["tutor_block_prompt_sha256"] = hashlib.sha256(pipeline.tutor_block_system.encode()).hexdigest()
    if hasattr(pipeline, "safe_alternative_system"):
        effective["safe_alternative_prompt_sha256"] = hashlib.sha256(pipeline.safe_alternative_system.encode()).hexdigest()
    if hasattr(pipeline, "review_system"):
        effective["answer_review_prompt_sha256"] = hashlib.sha256(pipeline.review_system.encode()).hexdigest()
    if hasattr(pipeline, "answer_choice_system"):
        effective["answer_choice_prompt_sha256"] = hashlib.sha256(pipeline.answer_choice_system.encode()).hexdigest()
    if hasattr(pipeline, "query_resolve_system"):
        effective["query_resolve_prompt_sha256"] = hashlib.sha256(pipeline.query_resolve_system.encode()).hexdigest()
    print("EXPERIMENT_CONFIG " + json.dumps(effective), flush=True)
    uvicorn.run("server:app", host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
