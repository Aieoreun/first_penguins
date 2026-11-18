"""요청 처리 흐름. 구조가 다른 흐름을 폴더로 나눠 두고 환경변수 PIPELINE으로 하나를 고른다.

  v1: 검색 → 입력 가드(호출 한 번) → 답변 생성 또는 출력 가드
  v2: 검색 → 입력 가드(셋으로 나눠 동시에) → 답변 생성 또는 출력 가드. 실험용 스위치가 있다
  v3: v2에 검증한 설정을 고정한 것 (다수결, 새 답변 프롬프트 등)
  v4: v3에서 다수결을 빼고(확인을 한 번씩만) 요청 전체의 마감 시간을 둔 것

검색·모델 호출·구절 찾기 같은 부품은 core/에 있고 모든 흐름이 같이 쓴다.
"""
import importlib
import os

DEFAULT = "v1"


def load(name: str = ""):
    """흐름 이름으로 Pipeline 클래스를 돌려준다."""
    name = name or os.environ.get("PIPELINE", DEFAULT)
    if name.startswith("jh_"):
        from pipelines.kimjaehak.config import VARIANTS, configure
        if name not in VARIANTS:
            raise ValueError(f"Unknown experiment pipeline: {name}")
        configure()
        if name in {"jh_r6_kiwi_bigram", "jh_r6_resolved_focused"}:
            module = importlib.import_module("pipelines.kimjaehak.round6_retrieval")
            return module.KiwiBigram if name == "jh_r6_kiwi_bigram" else module.ResolvedFocused
        if name in {"jh_r6_short_material", "jh_r6_material_nosection", "jh_r6_context_verify"}:
            module = importlib.import_module("pipelines.kimjaehak.round6")
            return {"jh_r6_short_material":module.ShortMaterial,
                    "jh_r6_material_nosection":module.NoSection,
                    "jh_r6_context_verify":module.ContextVerify}[name]
        if name in {"jh_r5_dual6", "jh_r5_dual_all"}:
            module = importlib.import_module("pipelines.kimjaehak.dual_granularity")
            return module.Six if name == "jh_r5_dual6" else module.All
        if name == "jh_r4_conditional_dense":
            return importlib.import_module("pipelines.kimjaehak.conditional_dense_source").Pipeline
        if name == "jh_r4_structured_block":
            return importlib.import_module("pipelines.kimjaehak.structured_block").Pipeline
        if name == "jh_r4_source_conversation_last":
            return importlib.import_module("pipelines.kimjaehak.source_conversation_last").SourceConversationLast
        if name == "jh_r4_material_budget":
            return importlib.import_module("pipelines.kimjaehak.material_budget").Pipeline
        if name == "jh_r4_source_minimal":
            return importlib.import_module("pipelines.kimjaehak.source_minimal").Pipeline
        if name == "jh_r4_source_disambiguated":
            return importlib.import_module("pipelines.kimjaehak.source_disambiguated").Pipeline
        if name == "jh_r4_source_id_only":
            return importlib.import_module("pipelines.kimjaehak.source_id_only").Pipeline
        if name == "jh_r4_budget_stream":
            return importlib.import_module("pipelines.kimjaehak.budget_stream").Pipeline
        if name == "jh_r4_source_stream":
            return importlib.import_module("pipelines.kimjaehak.source_stream").Pipeline
        if name == "jh_r4_last_dense":
            return importlib.import_module("pipelines.kimjaehak.last_dense_source").Pipeline
        if name == "jh_r4_normalized_tutor":
            return importlib.import_module("pipelines.kimjaehak.normalized_tutor").Pipeline
        if name == "jh_r3_safe_menu":
            return importlib.import_module("pipelines.kimjaehak.safe_menu").Pipeline
        if name == "jh_r3_safe_alternative":
            return importlib.import_module("pipelines.kimjaehak.safe_alternative").Pipeline
        if name == "jh_r3_material_vote3":
            return importlib.import_module("pipelines.kimjaehak.material_votes").Pipeline
        if name in {"jh_r3_extract_focused", "jh_r3_selected_focused", "jh_r3_grounded_focused", "jh_r3_quote_selected_focused", "jh_r3_expanded_focused"}:
            module = importlib.import_module("pipelines.kimjaehak.focused_excerpt")
            return {"jh_r3_extract_focused":module.Pipeline, "jh_r3_selected_focused":module.Selected,
                    "jh_r3_grounded_focused":module.Grounded, "jh_r3_quote_selected_focused":module.Quoted,
                    "jh_r3_expanded_focused":module.Expanded}[name]
        if name in {"jh_r3_tutor_other_boundary", "jh_r3_tutor_material_boundary"}:
            module = importlib.import_module("pipelines.kimjaehak.tutor_boundaries")
            return module.Other if name == "jh_r3_tutor_other_boundary" else module.Material
        if name in {"jh_r3_extract_union", "jh_r3_extract_union_tables"}:
            module = importlib.import_module("pipelines.kimjaehak.extractive_answer")
            return module.QueryUnion if name == "jh_r3_extract_union" else module.QueryUnionTables
        if name in {"jh_r3_extract_section", "jh_r3_extract_chunk"}:
            module = importlib.import_module("pipelines.kimjaehak.extractive_answer")
            return module.Section if name == "jh_r3_extract_section" else module.Chunk
        if name in {"jh_r3_material_extract", "jh_r3_gated_extract", "jh_r3_selected_extract", "jh_r3_grounded_extract"}:
            module = importlib.import_module("pipelines.kimjaehak.source_extract")
            return {"jh_r3_material_extract":module.MaterialSection, "jh_r3_gated_extract":module.GatedSection,
                    "jh_r3_selected_extract":module.SelectedSection, "jh_r3_grounded_extract":module.GroundedSelected}[name]
        if name == "jh_r3_material_source":
            return importlib.import_module("pipelines.kimjaehak.material_source").Pipeline
        if name == "jh_r3_dual_section90":
            return importlib.import_module("pipelines.kimjaehak.dual_section").Pipeline
        if name == "jh_r3_answer_support":
            return importlib.import_module("pipelines.kimjaehak.answer_support").Pipeline
        if name == "jh_r3_tutor_fit3000":
            return importlib.import_module("pipelines.kimjaehak.tutor_fitting").Pipeline
        if name == "jh_r3_dual_query":
            return importlib.import_module("pipelines.kimjaehak.dual_query").Pipeline
        if name == "jh_r3_section90":
            return importlib.import_module("pipelines.kimjaehak.tutor_section_gate").Pipeline
        if name == "jh_r3_query_gate":
            return importlib.import_module("pipelines.kimjaehak.query_gate").Pipeline
        if name in {"jh_r3_resolved_query", "jh_r3_resolved_minimal", "jh_r3_topic_select"}:
            module = importlib.import_module("pipelines.kimjaehak.resolved_query")
            if name == "jh_r3_topic_select":
                return module.TopicSelect
            return module.Minimal if name.endswith('_minimal') else module.Pipeline
        if name in {"jh_r3_quote_answer", "jh_r3_complete_answer", "jh_r3_checked_answer", "jh_r3_sentence_answer"}:
            module = importlib.import_module("pipelines.kimjaehak.answer_style")
            return {"jh_r3_quote_answer": module.Quote, "jh_r3_complete_answer": module.Complete,
                    "jh_r3_checked_answer": module.Checked, "jh_r3_sentence_answer": module.Sentence}[name]
        if name == "jh_r3_tutor_ranked":
            return importlib.import_module("pipelines.kimjaehak.tutor_ranked").Pipeline
        if name in {"jh_r3_answer_choice", "jh_r3_answer_pair", "jh_r3_answer_longest"}:
            module = importlib.import_module("pipelines.kimjaehak.answer_choice")
            if name == "jh_r3_answer_longest":
                return module.Longest
            return module.Pair if name.endswith("_pair") else module.Pipeline
        if name == "jh_r2_english_answer":
            return importlib.import_module("pipelines.kimjaehak.english_answer").Pipeline
        if name == "jh_r2_factual_scaffold":
            return importlib.import_module("pipelines.kimjaehak.factual_scaffold").Pipeline
        if name == "jh_r2_answer_review":
            return importlib.import_module("pipelines.kimjaehak.answer_review").Pipeline
        if name in {"jh_r2_tutor_split", "jh_r2_tutor_dense", "jh_r2_tutor_material10", "jh_r2_tutor_block"}:
            module = importlib.import_module("pipelines.kimjaehak.tutor_split")
            return {"jh_r2_tutor_split": module.Pipeline, "jh_r2_tutor_dense": module.DenseOnly,
                    "jh_r2_tutor_material10": module.MaterialOnly, "jh_r2_tutor_block": module.BlockOnly}[name]
        if name == "jh_r2_chat128":
            return importlib.import_module("pipelines.kimjaehak.chat_budget").Pipeline
        if name in {"jh_r2_ground2", "jh_r2_ground3"}:
            module = importlib.import_module("pipelines.kimjaehak.ground_sections")
            return module.Two if name.endswith("2") else module.Three
        if name in {"jh_r2_policy_review", "jh_r2_policy_explained", "jh_r2_policy_complied"}:
            module = importlib.import_module("pipelines.kimjaehak.policy_review")
            if name.endswith("_complied"):
                return module.Complied
            return module.Explained if name.endswith("_explained") else module.Pipeline
        if name == "jh_r2_source_choice":
            return importlib.import_module("pipelines.kimjaehak.source_choice").Pipeline
        if name in {"jh_r2_joint_input", "jh_r2_joint_gate98", "jh_r2_joint_ungated",
                    "jh_r2_joint_block", "jh_r2_joint_block_ungated"}:
            module = importlib.import_module("pipelines.kimjaehak.joint_input")
            return {"jh_r2_joint_input":module.Pipeline,"jh_r2_joint_gate98":module.Threshold98,
                    "jh_r2_joint_ungated":module.Ungated,"jh_r2_joint_block":module.BlockOnly,
                    "jh_r2_joint_block_ungated":module.BlockOnlyUngated}[name]
        if name in {"jh_r2_gate98", "jh_r2_gate995"}:
            module = importlib.import_module("pipelines.kimjaehak.conservative_gate")
            return module.Threshold98 if name == "jh_r2_gate98" else module.Threshold995
        if name == "jh_r2_candidate":
            return importlib.import_module("pipelines.kimjaehak.round2_candidate").Pipeline
        if name in {"jh_r2_input_examples", "jh_r2_dense_input", "jh_r2_dense_ungated", "jh_r2_dense98", "jh_r2_dense10"}:
            module = importlib.import_module("pipelines.kimjaehak.input_examples")
            if name == "jh_r2_dense10":
                return module.DenseTen
            if name == "jh_r2_dense98":
                return module.DenseConservative
            if name == "jh_r2_dense_ungated":
                return module.DenseUngated
            return module.Dense if name == "jh_r2_dense_input" else module.Lexical
        if name.startswith("jh_r2_"):
            module = importlib.import_module("pipelines.kimjaehak.round2")
            return {"jh_r2_sources": module.Sources, "jh_r2_rrf": module.Fusion,
                    "jh_r2_rrf_sources": module.FusionSources, "jh_r2_examples": module.Examples,
                    "jh_r2_rrf_examples": module.FusionExamples}[name]
        if name == "jh_baseline":
            return importlib.import_module("pipelines.v2.pipeline").Pipeline
        if name == "jh_v4_output":
            return importlib.import_module("pipelines.kimjaehak.v4_output").Pipeline
        if name == "jh_precise_output":
            return importlib.import_module("pipelines.kimjaehak.precise_output").Pipeline
        if name == "jh_calibrated":
            return importlib.import_module("pipelines.kimjaehak.calibrated").Pipeline
        if name == "jh_v4_safety":
            return importlib.import_module("pipelines.kimjaehak.v4_safety").Pipeline
        if name == "jh_voted":
            return importlib.import_module("pipelines.kimjaehak.voted").Pipeline
        if name == "jh_fewshot":
            return importlib.import_module("pipelines.kimjaehak.fewshot").Pipeline
        if name == "jh_reasoned":
            return importlib.import_module("pipelines.kimjaehak.reasoned").Pipeline
        if name == "jh_compact":
            return importlib.import_module("pipelines.kimjaehak.compact").Pipeline
        if name == "jh_explicit":
            return importlib.import_module("pipelines.kimjaehak.explicit").Pipeline
        if name == "jh_adaptive_explicit":
            return importlib.import_module("pipelines.kimjaehak.explicit").AdaptivePipeline
        if name == "jh_brief_material":
            return importlib.import_module("pipelines.kimjaehak.brief_material").Pipeline
        if name == "jh_concise_answer":
            return importlib.import_module("pipelines.kimjaehak.concise_answer").Pipeline
        if name == "jh_concise_fitted":
            return importlib.import_module("pipelines.kimjaehak.concise_answer").Fitted
        if name == "jh_evidence_answer":
            return importlib.import_module("pipelines.kimjaehak.evidence_answer").Pipeline
        if name == "jh_learned_harmful":
            return importlib.import_module("pipelines.kimjaehak.learned_harmful").Pipeline
        if name == "jh_plain_abstain":
            return importlib.import_module("pipelines.kimjaehak.plain_abstain").Pipeline
        if name == "jh_candidate":
            return importlib.import_module("pipelines.kimjaehak.candidate").Pipeline
        if name == "jh_candidate_stronger":
            return importlib.import_module("pipelines.kimjaehak.candidate").Stronger
        if name == "jh_brief_policy":
            return importlib.import_module("pipelines.kimjaehak.brief_policy").Pipeline
        if name == "jh_clear_policy":
            return importlib.import_module("pipelines.kimjaehak.brief_policy").ClearScaffold
        if name == "jh_medium_answer":
            return importlib.import_module("pipelines.kimjaehak.medium_answer").Pipeline
        if name in {"jh_voted_output", "jh_voted_output5"}:
            module = importlib.import_module("pipelines.kimjaehak.voted_output")
            return module.Five if name.endswith("5") else module.Pipeline
        if name in {"jh_uncertain_sources", "jh_resilient_sources", "jh_six_sources"}:
            module = importlib.import_module("pipelines.kimjaehak.uncertain_sources")
            return {"jh_uncertain_sources": module.Pipeline, "jh_resilient_sources": module.Resilient,
                    "jh_six_sources": module.Six}[name]
        if name in {"jh_context_policy", "jh_context_policy_voted"}:
            module = importlib.import_module("pipelines.kimjaehak.context_policy")
            return module.WithVotedOutput if name.endswith("_voted") else module.Pipeline
        if name in {"jh_learned_output", "jh_learned_output5", "jh_learned_output_stronger", "jh_guard_category"}:
            module = importlib.import_module("pipelines.kimjaehak.learned_output")
            return {"jh_learned_output":module.Pipeline, "jh_learned_output5":module.Consensus,
                    "jh_learned_output_stronger":module.Stronger, "jh_guard_category":module.GuardCategory}[name]
        if name in {"jh_context_grounding", "jh_context_grounding_medium"}:
            module = importlib.import_module("pipelines.kimjaehak.context_grounding")
            return module.Reasoned if name.endswith("_medium") else module.Pipeline
        if name == "jh_hybrid_output":
            return importlib.import_module("pipelines.kimjaehak.hybrid_output").Pipeline
        if name == "jh_guard_fitted_span99":
            return importlib.import_module("pipelines.kimjaehak.span_decision").Pipeline
        if name in {"jh_guard_fitted_hybrid", "jh_guard_fitted_fast", "jh_guard_fitted_contextfast"}:
            module = importlib.import_module("pipelines.kimjaehak.guard_fitted")
            if name.endswith("_contextfast"):
                return module.ContextFast
            return module.Fast if name.endswith("_fast") else module.Pipeline
        if name in {"jh_refusal_review", "jh_hybrid_refusal", "jh_hybrid_refusal_broad"}:
            module = importlib.import_module("pipelines.kimjaehak.refusal_review")
            if name == "jh_hybrid_refusal_broad":
                return module.BroadPipeline
            return module.HybridPipeline if name == "jh_hybrid_refusal" else module.Pipeline
        if name in {"jh_context_output", "jh_context_output3"}:
            module = importlib.import_module("pipelines.kimjaehak.context_output")
            return module.ThreeSections if name.endswith("3") else module.Pipeline
        if name == "jh_reasoned_tie":
            return importlib.import_module("pipelines.kimjaehak.reasoned_tie").Pipeline
        if name == "jh_learned_gate":
            return importlib.import_module("pipelines.kimjaehak.learned_gate").Pipeline
        if name == "jh_learned_gate98":
            return importlib.import_module("pipelines.kimjaehak.learned_gate").Relaxed
        if name == "jh_learned_gate95":
            return importlib.import_module("pipelines.kimjaehak.learned_gate").Broad
        if name in {"jh_learned_sections", "jh_learned_sections_all", "jh_learned_sections_tutor", "jh_learned_sections_tutor95"}:
            module = importlib.import_module("pipelines.kimjaehak.learned_sections")
            return {"jh_learned_sections":module.Pipeline, "jh_learned_sections_all":module.WholeKB,
                    "jh_learned_sections_tutor":module.TutorOnly,
                    "jh_learned_sections_tutor95":module.WideTutor}[name]
        if name in {"jh_stronger95", "jh_stronger98"}:
            module = importlib.import_module("pipelines.kimjaehak.learned_sections")
            return module.Stronger if name.endswith("95") else module.StrongerConservative
        if name in {"jh_kb_expand", "jh_kb_expand_fitted"}:
            module = importlib.import_module("pipelines.kimjaehak.kb_expanded")
            return module.Fitted if name.endswith("_fitted") else module.Pipeline
        if name == "jh_intent":
            return importlib.import_module("pipelines.kimjaehak.intent").Pipeline
        if name == "jh_diverse":
            return importlib.import_module("pipelines.kimjaehak.diverse").Pipeline
        if name == "jh_multiground":
            return importlib.import_module("pipelines.kimjaehak.multiground").Pipeline
        if name == "jh_material_review":
            return importlib.import_module("pipelines.kimjaehak.material_review").Pipeline
        if name == "jh_block5":
            return importlib.import_module("pipelines.kimjaehak.block5").Pipeline
        if name == "jh_reconsider":
            return importlib.import_module("pipelines.kimjaehak.reconsider").Pipeline
        if name == "jh_unanimous":
            return importlib.import_module("pipelines.kimjaehak.unanimous").Pipeline
        if name == "jh_token_output":
            return importlib.import_module("pipelines.kimjaehak.token_output").Pipeline
        if name in {"jh_adaptive", "jh_adaptive_unanimous"}:
            module = importlib.import_module("pipelines.kimjaehak.adaptive")
            return module.Unanimous if name.endswith("_unanimous") else module.Pipeline
        if name == "jh_medium_output":
            return importlib.import_module("pipelines.kimjaehak.medium_output").Pipeline
        if name in {"jh_thin_material", "jh_adaptive_thin"}:
            module = importlib.import_module("pipelines.kimjaehak.thin_material")
            return module.AdaptivePipeline if name == "jh_adaptive_thin" else module.Pipeline
        if name.startswith("jh_guided_"):
            module = importlib.import_module("pipelines.kimjaehak.guided")
            return {"jh_guided_input": module.GuidedInput,
                    "jh_guided_output": module.GuidedOutput,
                    "jh_guided_both": module.GuidedBoth,
                    "jh_guided_focus": module.FocusInput}[name]
        module = importlib.import_module("pipelines.kimjaehak.pipeline")
        return {"jh_peace_input": module.PeaceInput,
                "jh_peace_output": module.PeaceOutput,
                "jh_peace_both": module.PeaceBoth}[name]
    return importlib.import_module(f"pipelines.{name}.pipeline").Pipeline
