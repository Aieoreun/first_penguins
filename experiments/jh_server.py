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
    from dotenv import load_dotenv
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
    load_dotenv(ROOT / ".env", override=True)
    if args.embedding_mode == "off":
        for key in ("EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_API_KEY"):
            os.environ[key] = ""
    os.environ["PIPELINE"] = args.variant
    for key in ("TEAM_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if not os.environ.get(key):
            raise SystemExit(f"Missing setting: {key}")
    from pipelines import load
    pipeline = load(args.variant)
    if args.request_slots is not None:
        if not hasattr(pipeline, "request_slots"):
            parser.error("This pipeline has no configurable request semaphore")
        pipeline.request_slots = args.request_slots
    fields = ("answer_effort", "answer_max_tokens", "votes", "seek_first", "safe_by_rule",
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
              "tutor_safe_alternative", "tutor_safe_menu", "tutor_category_normalization", "source_last_dense", "source_last_dense_timeout", "source_last_dense_sections")
    effective = {key: getattr(pipeline, key, None) for key in fields}
    effective["class"] = pipeline.__module__ + "." + pipeline.__name__
    effective["embedding_mode"] = args.embedding_mode
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
    if hasattr(pipeline, "policy_system"):
        effective["policy_prompt_sha256"] = hashlib.sha256(pipeline.policy_system.encode()).hexdigest()
    if hasattr(pipeline, "source_choice_system"):
        effective["source_choice_prompt_sha256"] = hashlib.sha256(pipeline.source_choice_system.encode()).hexdigest()
    if hasattr(pipeline, "source_extract_system"):
        effective["source_extract_prompt_sha256"] = hashlib.sha256(pipeline.source_extract_system.encode()).hexdigest()
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
