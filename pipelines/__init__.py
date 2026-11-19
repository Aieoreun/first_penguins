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
        if name in {'jh_r7_tax_scope', 'jh_r7_tax_concrete'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_tax_scope')
            return module.ExplicitTaxScope if name == 'jh_r7_tax_scope' else module.TaxScopedConcrete
        if name == 'jh_r7_binary_guard':
            return importlib.import_module('pipelines.kimjaehak.round7_binary_guard').BinaryGuard
        if name == 'jh_r7_calc_review':
            return importlib.import_module('pipelines.kimjaehak.round7_calculator_review').ReviewedCalculator
        if name in {'jh_r7_numeric_verdict', 'jh_r7_numeric_compact', 'jh_r7_numeric_direct'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_numeric_combined')
            return {'jh_r7_numeric_verdict': module.NumericMaterialVerdict, 'jh_r7_numeric_compact': module.NumericVerdictCompact, 'jh_r7_numeric_direct': module.NumericVerdictDirect}[name]
        if name == 'jh_r7_material_verdict':
            return importlib.import_module('pipelines.kimjaehak.round7_material_verdict').MaterialVerdict
        if name in {'jh_r7_calc_noquote', 'jh_r7_calc_compact'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_compact_calculator')
            return module.NoQuoteCalculator if name == 'jh_r7_calc_noquote' else module.CompactCalculator
        if name == 'jh_r7_early_consensus':
            return importlib.import_module('pipelines.kimjaehak.round7_early_consensus').EarlyConsensus
        if name in {'jh_r7_task_examples1', 'jh_r7_task_examples3'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_task_examples')
            return module.TaskExampleOne if name == 'jh_r7_task_examples1' else module.TaskExamples
        if name == 'jh_r7_adaptive3':
            return importlib.import_module('pipelines.kimjaehak.round7_adaptive_ground').AdaptiveThree
        if name in {'jh_r7_structured_calc', 'jh_r7_scope_structured'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_structured_calculator')
            return module.StructuredCalculator if name == 'jh_r7_structured_calc' else module.ScopedStructuredCalculator
        if name in {'jh_r7_examples1', 'jh_r7_examples3'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_examples')
            return module.RetrievedOne if name == 'jh_r7_examples1' else module.RetrievedExamples
        if name in {'jh_r7_calculator', 'jh_r7_material_calculator'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_calculator')
            return module.CalculatorApply if name == 'jh_r7_calculator' else module.MaterialCalculator
        if name in {'jh_r7_minimal_span', 'jh_r7_adaptive_span'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_span_prompt')
            return module.MinimalClause if name == 'jh_r7_minimal_span' else module.AdaptiveMinimal
        if name == 'jh_r7_blend_adaptive':
            return importlib.import_module('pipelines.kimjaehak.round7_combined').BlendedAdaptive
        if name in {'jh_r7_cancel_guard', 'jh_r7_cancel_tutor', 'jh_r7_cancel_both'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_material_cancel')
            return {'jh_r7_cancel_guard':module.CancelGuardMaterial, 'jh_r7_cancel_tutor':module.CancelTutorMaterial, 'jh_r7_cancel_both':module.CancelBothMaterial}[name]
        if name in {'jh_r7_material_scope', 'jh_r7_apply_scoped'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_material_scope')
            return module.MaterialScope if name == 'jh_r7_material_scope' else module.ScopedConcrete
        if name in {'jh_r7_ground3', 'jh_r7_ground_adaptive'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_adaptive_ground')
            return module.GroundThree if name == 'jh_r7_ground3' else module.AdaptiveGround
        if name in {'jh_r7_ground_input', 'jh_r7_ground_union'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_ground_source')
            return module.InputGroundSource if name == 'jh_r7_ground_input' else module.UnionGroundSource
        if name in {'jh_r7_apply_concrete', 'jh_r7_apply_joint'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_answer')
            return module.ConcreteApply if name == 'jh_r7_apply_concrete' else module.JointConcreteApply
        if name in {'jh_r7_apply_numeric', 'jh_r7_apply_all'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_answer')
            return module.NumericApply if name == 'jh_r7_apply_numeric' else module.AllApply
        if name in {'jh_r7_batchconcat', 'jh_r7_batchblend'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_batch')
            return module.BatchedConcat if name == 'jh_r7_batchconcat' else module.BatchedBlend
        if name in {'jh_r7_concat', 'jh_r7_blend', 'jh_r7_fastblend', 'jh_r7_llm_input'}:
            module = importlib.import_module('pipelines.kimjaehak.round7_hybrid')
            return {'jh_r7_concat': module.ConcatGate, 'jh_r7_blend': module.BlendGate,
                    'jh_r7_fastblend': module.FastBlendGate, 'jh_r7_llm_input': module.CommonFallback}[name]
        if name in {'jh_r7_boundary', 'jh_r7_verdict'}:
            module = importlib.import_module("pipelines.kimjaehak.round7_guard")
            return {'jh_r7_boundary': module.BoundaryGate, 'jh_r7_verdict': module.VerdictFirst}[name]
        if name in {'jh_r7_scaffold', 'jh_r7_other', 'jh_r7_product'}:
            module = importlib.import_module("pipelines.kimjaehak.round7_policy")
            return {'jh_r7_product': module.ProductContext, 'jh_r7_other': module.OtherBoundary, 'jh_r7_scaffold': module.ScaffoldGuide}[name]
        if name in {'jh_re_c_groundlow2k', 'jh_re_c_input2k', 'jh_re_c_ground2k', 'jh_re_a_input', 'jh_re_a_source', 'jh_re_c_ground', 'jh_re_c_input', 'jh_re_c_safety', 'jh_re_a_answer'}:
            module = importlib.import_module("pipelines.kimjaehak.reasoning_ablation")
            return {'jh_re_c_groundlow2k': module.GuardGroundLowBudget, 'jh_re_c_input2k': module.GuardInputMediumBudget, 'jh_re_c_ground2k': module.GuardGroundMediumBudget, 'jh_re_c_input': module.GuardInputMedium, 'jh_re_c_ground': module.GuardGroundMedium, 'jh_re_c_safety': module.GuardSafetyMedium, 'jh_re_a_input': module.TutorInputMedium, 'jh_re_a_source': module.TutorSourceMedium, 'jh_re_a_answer': module.TutorAnswerMedium}[name]
        if name in {"jh_np_parallel", "jh_np_timeouts", "jh_np_peace_block", "jh_np_unlearned", "jh_np_input_budgets", "jh_np_block_only"}:
            module = importlib.import_module("pipelines.kimjaehak.neural_peace_guard")
            return {"jh_np_parallel": module.ParallelSafety, "jh_np_timeouts": module.CheckBudgets,
                    "jh_np_peace_block": module.PeaceBlock, "jh_np_unlearned": module.UnlearnedInput,
                    "jh_np_input_budgets": module.InputBudgets, "jh_np_block_only": module.BlockOnly}[name]
        if name in {"jh_tm_selective", "jh_tm_replace"}:
            module = importlib.import_module("pipelines.kimjaehak.tutor_mlp")
            return module.Selective if name == "jh_tm_selective" else module.Replace
        if name in {"jh_cm_char_lr50", "jh_cm_dense_lr50", "jh_cm_char_mlp50", "jh_cm_dense_mlp50"}:
            module = importlib.import_module("pipelines.kimjaehak.classifier_matrix")
            return {"jh_cm_char_lr50": module.CharLR50, "jh_cm_dense_lr50": module.DenseLR50,
                    "jh_cm_char_mlp50": module.CharMLP50, "jh_cm_dense_mlp50": module.DenseMLP50}[name]
        if name in {"jh_cm_char_lr", "jh_cm_dense_lr", "jh_cm_char_mlp", "jh_cm_dense_mlp"}:
            module = importlib.import_module("pipelines.kimjaehak.classifier_matrix")
            return {"jh_cm_char_lr": module.CharLR, "jh_cm_dense_lr": module.DenseLR,
                    "jh_cm_char_mlp": module.CharMLP, "jh_cm_dense_mlp": module.DenseMLP}[name]
        if name == "jh_pg_allow_safety":
            return importlib.import_module("pipelines.kimjaehak.peace_allow_safety").AllowSafety
        if name in {"jh_pg_fast_cancel", "jh_pg_cancel_requested"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_cancel_ground")
            return module.FastCancelBlock if name == "jh_pg_fast_cancel" else module.FastCancelRequested
        if name in {"jh_pg_current_gate90", "jh_pg_fast_gate90"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_gate_threshold")
            return module.CurrentGate90 if name == "jh_pg_current_gate90" else module.FastGate90
        if name in {"jh_pg_current_review", "jh_pg_fast_review", "jh_pg_current_review80", "jh_pg_fast_review80"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_span_review")
            return {"jh_pg_current_review": module.CurrentReview, "jh_pg_fast_review": module.FastReview,
                    "jh_pg_current_review80": module.CurrentReview80, "jh_pg_fast_review80": module.FastReview80}[name]
        if name in {"jh_pg_fast_gate", "jh_pg_parallel_input", "jh_pg_fast_parallel", "jh_pg_fast_requested"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_guard_scheduling")
            return {"jh_pg_fast_gate": module.FastGate, "jh_pg_parallel_input": module.ParallelInput,
                    "jh_pg_fast_parallel": module.FastParallelInput, "jh_pg_fast_requested": module.FastRequestedParallel}[name]
        if name in {"jh_pg_ground3", "jh_pg_medium3", "jh_pg_current3", "jh_pg_fast3"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_ground_budget")
            return {"jh_pg_ground3": module.LowThree, "jh_pg_medium3": module.MediumThree,
                    "jh_pg_current3": module.CurrentThree, "jh_pg_fast3": module.FastParallelThree}[name]
        if name in {"jh_pg_current_span80", "jh_pg_routed_span80", "jh_pg_requested_span80"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_span_variants")
            return {"jh_pg_current_span80": module.CurrentSpan80, "jh_pg_routed_span80": module.RoutedSpan80,
                    "jh_pg_requested_span80": module.RoutedRequestedSpan80}[name]
        if name == "jh_pg_fusion":
            return importlib.import_module("pipelines.kimjaehak.peace_guard_fusion").Pipeline
        if name == "jh_pg_requested":
            return importlib.import_module("pipelines.kimjaehak.peace_guard_variants").RequestedAction
        if name in {"jh_pg_action", "jh_pg_context"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_safety_variants")
            return module.ActionSafety if name == "jh_pg_action" else module.ContextSafety
        if name in {"jh_pg_conditioned", "jh_pg_requested_conditioned"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_conditioned_safety")
            return module.ConditionedSafety if name == "jh_pg_conditioned" else module.RequestedConditioned
        if name in {"jh_pg_relation", "jh_pg_provenance"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_ground_variants")
            return module.RelationGrounding if name == "jh_pg_relation" else module.ProvenanceGrounding
        if name in {"jh_pg_vote3", "jh_pg_conditioned_vote3"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_voted_safety")
            return module.VotedSafety if name == "jh_pg_vote3" else module.VotedConditioned
        if name in {"jh_pg_routed", "jh_pg_routed_requested", "jh_pg_routed_relation", "jh_pg_routed_provenance"}:
            module = importlib.import_module("pipelines.kimjaehak.peace_routed_safety")
            return {"jh_pg_routed": module.RoutedSafety, "jh_pg_routed_requested": module.RoutedRequested,
                    "jh_pg_routed_relation": module.RoutedRelation, "jh_pg_routed_provenance": module.RoutedProvenance}[name]
        if name in {"jh_r6_evidence_first", "jh_r6_source_vote3"}:
            module = importlib.import_module("pipelines.kimjaehak.round6_source")
            return module.EvidenceFirst if name == "jh_r6_evidence_first" else module.VoteThree
        if name == "jh_r6_product_context":
            return importlib.import_module("pipelines.kimjaehak.round6_product_context").Pipeline
        if name == "jh_r6_source_conditioned_gate":
            return importlib.import_module("pipelines.kimjaehak.round6_source_conditioned_gate").Pipeline
        if name in {"jh_r6_source_support", "jh_r6_source_support12"}:
            module = importlib.import_module("pipelines.kimjaehak.round6_source_support")
            return module.Pipeline if name == "jh_r6_source_support" else module.Budget
        if name in {"jh_r6_eager_source", "jh_r6_eager_source12"}:
            module = importlib.import_module("pipelines.kimjaehak.round6_eager_source")
            return module.Eager if name == "jh_r6_eager_source" else module.Budget
        if name == "jh_r6_joint_fact":
            return importlib.import_module("pipelines.kimjaehak.round6_joint_fact").Pipeline
        if name == "jh_r6_resolved_material_hint":
            return importlib.import_module("pipelines.kimjaehak.round6_material_hint").Pipeline
        if name == "jh_r6_normal94_sparse":
            return importlib.import_module("pipelines.kimjaehak.round6_normal_sparse").Pipeline
        if name == "jh_r6_joint_quote":
            return importlib.import_module("pipelines.kimjaehak.round6_joint_quote").Pipeline
        if name == "jh_r6_trusted_sparse":
            return importlib.import_module("pipelines.kimjaehak.round6_trusted_sparse").Pipeline
        if name == "jh_r6_joint_evidence":
            return importlib.import_module("pipelines.kimjaehak.round6_joint_evidence").Pipeline
        if name == "jh_r6_normal_gate94":
            return importlib.import_module("pipelines.kimjaehak.round6_normal_gate").Pipeline
        if name in {"jh_r6_source_budget12", "jh_r6_source_reason12"}:
            module = importlib.import_module("pipelines.kimjaehak.round6_source_budget")
            return module.Budget if name == "jh_r6_source_budget12" else module.Reason
        if name == "jh_r6_resolved_source_hint":
            return importlib.import_module("pipelines.kimjaehak.round6_query_hint").SourceHint
        if name == "jh_r6_computable_material":
            return importlib.import_module("pipelines.kimjaehak.round6_material").Computable
        if name == "jh_r6_topic_focused":
            return importlib.import_module("pipelines.kimjaehak.round6_retrieval").TopicFocused
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
