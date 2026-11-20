"""Compare pinned public guard diagnostics and an exact Peace snapshot.

No production module imports this development recorder. Snapshot source files,
the official evaluator and both servers remain unmodified. --prepare-only never
starts a server; --collect-only checkpoints HTTP before separate scoring.
"""
import argparse
from collections import Counter
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import jh_compare as shared
from experiments import jh_guard_fusion_datasets as datasets

PEACE_COMMIT = '3c64079eb1af14170b3bee4578ac6f916bec433e'
CANDIDATES = {
    'r8_failure_input': ('jh_r8_failure_input', 'pipelines.kimjaehak.round8_failure_fallback.FailureInput'),
    'r8_failure_ground': ('jh_r8_failure_ground', 'pipelines.kimjaehak.round8_failure_fallback.FailureGround'),
    'r8_failure_both': ('jh_r8_failure_both', 'pipelines.kimjaehak.round8_failure_fallback.FailureBoth'),

    'r8_risk10': ('jh_r8_risk10', 'pipelines.kimjaehak.round8_risk_safety.RiskSafety'),
    'r8_risk50': ('jh_r8_risk50', 'pipelines.kimjaehak.round8_risk_safety.RiskSafety50'),

    'r8_binary_boundary': ('jh_r8_binary_boundary', 'pipelines.kimjaehak.round8_category.BinaryBoundary'),
    'r8_binary_boundary_budgets': ('jh_r8_binary_boundary_budgets', 'pipelines.kimjaehak.round8_category.BinaryBoundaryBudgets'),
    'r8_binary_dense': ('jh_r8_binary_dense', 'pipelines.kimjaehak.round8_category.BinaryDense'),

    'r8_binary_parallel': ('jh_r8_binary_parallel', 'pipelines.kimjaehak.round8_guard.BinaryParallelSafety'),
    'r8_binary_timeouts': ('jh_r8_binary_timeouts', 'pipelines.kimjaehak.round8_guard.BinaryCheckBudgets'),
    'r8_binary_inputbudgets': ('jh_r8_binary_inputbudgets', 'pipelines.kimjaehak.round8_guard.BinaryInputBudgets'),

    'r8_value_prompt': ('jh_r8_value_prompt', 'pipelines.kimjaehak.round8_numeric.ValuePrompt'),
    'r8_value_anchored': ('jh_r8_value_anchored', 'pipelines.kimjaehak.round8_numeric.ValueAnchored'),
    'r8_binary_value_prompt': ('jh_r8_binary_value_prompt', 'pipelines.kimjaehak.round8_numeric.BinaryValuePrompt'),
    'r8_binary_value_anchored': ('jh_r8_binary_value_anchored', 'pipelines.kimjaehak.round8_numeric.BinaryValueAnchored'),

    'r7_tax_budget12': ('jh_r7_tax_budget12', 'pipelines.kimjaehak.round7_tax_scope.TaxBudget12'),
    'r7_binary_tutor': ('jh_r7_binary_tutor', 'pipelines.kimjaehak.round7_binary_tutor.BinaryTutor'),
    'r7_binary_tutor12': ('jh_r7_binary_tutor12', 'pipelines.kimjaehak.round7_binary_tutor.BinaryTutor12'),
    'r7_tax_scope': ('jh_r7_tax_scope', 'pipelines.kimjaehak.round7_tax_scope.ExplicitTaxScope'),
    'r7_tax_concrete': ('jh_r7_tax_concrete', 'pipelines.kimjaehak.round7_tax_scope.TaxScopedConcrete'),
    'r7_binary_guard75': ('jh_r7_binary_guard75', 'pipelines.kimjaehak.round7_binary_guard.BinaryGuard75'),
    'r7_binary_guard': ('jh_r7_binary_guard', 'pipelines.kimjaehak.round7_binary_guard.BinaryGuard'),
    'r7_calc_review': ('jh_r7_calc_review', 'pipelines.kimjaehak.round7_calculator_review.ReviewedCalculator'),
    'r7_numeric_verdict': ('jh_r7_numeric_verdict', 'pipelines.kimjaehak.round7_numeric_combined.NumericMaterialVerdict'),
    'r7_numeric_compact': ('jh_r7_numeric_compact', 'pipelines.kimjaehak.round7_numeric_combined.NumericVerdictCompact'),
    'r7_numeric_direct': ('jh_r7_numeric_direct', 'pipelines.kimjaehak.round7_numeric_combined.NumericVerdictDirect'),
    'r7_material_verdict': ('jh_r7_material_verdict', 'pipelines.kimjaehak.round7_material_verdict.MaterialVerdict'),
    'r7_calc_noquote': ('jh_r7_calc_noquote', 'pipelines.kimjaehak.round7_compact_calculator.NoQuoteCalculator'),
    'r7_calc_compact': ('jh_r7_calc_compact', 'pipelines.kimjaehak.round7_compact_calculator.CompactCalculator'),
    'r7_early_consensus': ('jh_r7_early_consensus', 'pipelines.kimjaehak.round7_early_consensus.EarlyConsensus'),
    'r7_task_examples1': ('jh_r7_task_examples1', 'pipelines.kimjaehak.round7_task_examples.TaskExampleOne'),
    'r7_task_examples3': ('jh_r7_task_examples3', 'pipelines.kimjaehak.round7_task_examples.TaskExamples'),
    'r7_adaptive3': ('jh_r7_adaptive3', 'pipelines.kimjaehak.round7_adaptive_ground.AdaptiveThree'),
    'r7_structured_calc': ('jh_r7_structured_calc', 'pipelines.kimjaehak.round7_structured_calculator.StructuredCalculator'),
    'r7_scope_structured': ('jh_r7_scope_structured', 'pipelines.kimjaehak.round7_structured_calculator.ScopedStructuredCalculator'),
    'r7_examples1': ('jh_r7_examples1', 'pipelines.kimjaehak.round7_examples.RetrievedOne'),
    'r7_examples3': ('jh_r7_examples3', 'pipelines.kimjaehak.round7_examples.RetrievedExamples'),
    'r7_material_calculator': ('jh_r7_material_calculator', 'pipelines.kimjaehak.round7_calculator.MaterialCalculator'),
    'r7_calculator': ('jh_r7_calculator', 'pipelines.kimjaehak.round7_calculator.CalculatorApply'),
    'r7_minimal_span': ('jh_r7_minimal_span', 'pipelines.kimjaehak.round7_span_prompt.MinimalClause'),
    'r7_adaptive_span': ('jh_r7_adaptive_span', 'pipelines.kimjaehak.round7_span_prompt.AdaptiveMinimal'),
    'r7_apply_numeric': ('jh_r7_apply_numeric', 'pipelines.kimjaehak.round7_answer.NumericApply'),
    'r7_apply_all': ('jh_r7_apply_all', 'pipelines.kimjaehak.round7_answer.AllApply'),
    'r7_ground_input': ('jh_r7_ground_input', 'pipelines.kimjaehak.round7_ground_source.InputGroundSource'),
    'r7_ground_union': ('jh_r7_ground_union', 'pipelines.kimjaehak.round7_ground_source.UnionGroundSource'),
    'r7_apply_concrete': ('jh_r7_apply_concrete', 'pipelines.kimjaehak.round7_answer.ConcreteApply'),
    'r7_apply_joint': ('jh_r7_apply_joint', 'pipelines.kimjaehak.round7_answer.JointConcreteApply'),
    'r7_ground3': ('jh_r7_ground3', 'pipelines.kimjaehak.round7_adaptive_ground.GroundThree'),
    'r7_ground_adaptive': ('jh_r7_ground_adaptive', 'pipelines.kimjaehak.round7_adaptive_ground.AdaptiveGround'),
    'r7_blend_adaptive': ('jh_r7_blend_adaptive', 'pipelines.kimjaehak.round7_combined.BlendedAdaptive'),
    'r7_cancel_guard': ('jh_r7_cancel_guard', 'pipelines.kimjaehak.round7_material_cancel.CancelGuardMaterial'),
    'r7_cancel_tutor': ('jh_r7_cancel_tutor', 'pipelines.kimjaehak.round7_material_cancel.CancelTutorMaterial'),
    'r7_cancel_both': ('jh_r7_cancel_both', 'pipelines.kimjaehak.round7_material_cancel.CancelBothMaterial'),
    'r7_material_scope': ('jh_r7_material_scope', 'pipelines.kimjaehak.round7_material_scope.MaterialScope'),
    'r7_apply_scoped': ('jh_r7_apply_scoped', 'pipelines.kimjaehak.round7_material_scope.ScopedConcrete'),
    'r7_llm_input': ('jh_r7_llm_input', 'pipelines.kimjaehak.round7_hybrid.CommonFallback'),
    'r7_batchconcat': ('jh_r7_batchconcat', 'pipelines.kimjaehak.round7_batch.BatchedConcat'),
    'r7_batchblend': ('jh_r7_batchblend', 'pipelines.kimjaehak.round7_batch.BatchedBlend'),
    'r7_concat': ('jh_r7_concat', 'pipelines.kimjaehak.round7_hybrid.ConcatGate'),
    'r7_blend': ('jh_r7_blend', 'pipelines.kimjaehak.round7_hybrid.BlendGate'),
    'r7_fastblend': ('jh_r7_fastblend', 'pipelines.kimjaehak.round7_hybrid.FastBlendGate'),
    'r7_boundary': ('jh_r7_boundary', 'pipelines.kimjaehak.round7_guard.BoundaryGate'),
    'r7_verdict': ('jh_r7_verdict', 'pipelines.kimjaehak.round7_guard.VerdictFirst'),

    'r7_product': ('jh_r7_product', 'pipelines.kimjaehak.round7_policy.ProductContext'),
    'r7_other': ('jh_r7_other', 'pipelines.kimjaehak.round7_policy.OtherBoundary'),
    'r7_scaffold': ('jh_r7_scaffold', 'pipelines.kimjaehak.round7_policy.ScaffoldGuide'),

    're_c_groundlow2k': ('jh_re_c_groundlow2k', 'pipelines.kimjaehak.reasoning_ablation.GuardGroundLowBudget'),
    're_c_input2k': ('jh_re_c_input2k', 'pipelines.kimjaehak.reasoning_ablation.GuardInputMediumBudget'),
    're_c_ground2k': ('jh_re_c_ground2k', 'pipelines.kimjaehak.reasoning_ablation.GuardGroundMediumBudget'),
    're_c_input': ('jh_re_c_input', 'pipelines.kimjaehak.reasoning_ablation.GuardInputMedium'),
    're_c_ground': ('jh_re_c_ground', 'pipelines.kimjaehak.reasoning_ablation.GuardGroundMedium'),
    're_c_safety': ('jh_re_c_safety', 'pipelines.kimjaehak.reasoning_ablation.GuardSafetyMedium'),
    're_a_input': ('jh_re_a_input', 'pipelines.kimjaehak.reasoning_ablation.TutorInputMedium'),
    're_a_source': ('jh_re_a_source', 'pipelines.kimjaehak.reasoning_ablation.TutorSourceMedium'),
    're_a_answer': ('jh_re_a_answer', 'pipelines.kimjaehak.reasoning_ablation.TutorAnswerMedium'),

    'np_input_budgets': ('jh_np_input_budgets', 'pipelines.kimjaehak.neural_peace_guard.InputBudgets'),
    'np_block_only': ('jh_np_block_only', 'pipelines.kimjaehak.neural_peace_guard.BlockOnly'),
    'np_parallel': ('jh_np_parallel', 'pipelines.kimjaehak.neural_peace_guard.ParallelSafety'),
    'np_timeouts': ('jh_np_timeouts', 'pipelines.kimjaehak.neural_peace_guard.CheckBudgets'),
    'np_peace_block': ('jh_np_peace_block', 'pipelines.kimjaehak.neural_peace_guard.PeaceBlock'),
    'np_unlearned': ('jh_np_unlearned', 'pipelines.kimjaehak.neural_peace_guard.UnlearnedInput'),
    'tm_selective': ('jh_tm_selective', 'pipelines.kimjaehak.tutor_mlp.Selective'),
    'tm_replace': ('jh_tm_replace', 'pipelines.kimjaehak.tutor_mlp.Replace'),
    'cm_char_lr50': ('jh_cm_char_lr50', 'pipelines.kimjaehak.classifier_matrix.CharLR50'),
    'cm_dense_lr50': ('jh_cm_dense_lr50', 'pipelines.kimjaehak.classifier_matrix.DenseLR50'),
    'cm_char_mlp50': ('jh_cm_char_mlp50', 'pipelines.kimjaehak.classifier_matrix.CharMLP50'),
    'cm_dense_mlp50': ('jh_cm_dense_mlp50', 'pipelines.kimjaehak.classifier_matrix.DenseMLP50'),
    'cm_char_lr': ('jh_cm_char_lr', 'pipelines.kimjaehak.classifier_matrix.CharLR'),
    'cm_dense_lr': ('jh_cm_dense_lr', 'pipelines.kimjaehak.classifier_matrix.DenseLR'),
    'cm_char_mlp': ('jh_cm_char_mlp', 'pipelines.kimjaehak.classifier_matrix.CharMLP'),
    'cm_dense_mlp': ('jh_cm_dense_mlp', 'pipelines.kimjaehak.classifier_matrix.DenseMLP'),
    'allow_safety': ('jh_pg_allow_safety', 'pipelines.kimjaehak.peace_allow_safety.AllowSafety'),
    'cancel_requested': ('jh_pg_cancel_requested', 'pipelines.kimjaehak.peace_cancel_ground.FastCancelRequested'),
    'fast_cancel': ('jh_pg_fast_cancel', 'pipelines.kimjaehak.peace_cancel_ground.FastCancelBlock'),
    'current_gate90': ('jh_pg_current_gate90', 'pipelines.kimjaehak.peace_gate_threshold.CurrentGate90'),
    'fast_gate90': ('jh_pg_fast_gate90', 'pipelines.kimjaehak.peace_gate_threshold.FastGate90'),
    'current_review': ('jh_pg_current_review', 'pipelines.kimjaehak.peace_span_review.CurrentReview'),
    'fast_review': ('jh_pg_fast_review', 'pipelines.kimjaehak.peace_span_review.FastReview'),
    'current_review80': ('jh_pg_current_review80', 'pipelines.kimjaehak.peace_span_review.CurrentReview80'),
    'fast_review80': ('jh_pg_fast_review80', 'pipelines.kimjaehak.peace_span_review.FastReview80'),
    'current3': ('jh_pg_current3', 'pipelines.kimjaehak.peace_ground_budget.CurrentThree'),
    'fast3': ('jh_pg_fast3', 'pipelines.kimjaehak.peace_ground_budget.FastParallelThree'),
    'fast_gate': ('jh_pg_fast_gate', 'pipelines.kimjaehak.peace_guard_scheduling.FastGate'),
    'parallel_input': ('jh_pg_parallel_input', 'pipelines.kimjaehak.peace_guard_scheduling.ParallelInput'),
    'fast_parallel': ('jh_pg_fast_parallel', 'pipelines.kimjaehak.peace_guard_scheduling.FastParallelInput'),
    'fast_requested': ('jh_pg_fast_requested', 'pipelines.kimjaehak.peace_guard_scheduling.FastRequestedParallel'),
    'ground3': ('jh_pg_ground3', 'pipelines.kimjaehak.peace_ground_budget.LowThree'),
    'medium3': ('jh_pg_medium3', 'pipelines.kimjaehak.peace_ground_budget.MediumThree'),
    'current_span80': ('jh_pg_current_span80', 'pipelines.kimjaehak.peace_span_variants.CurrentSpan80'),
    'routed_span80': ('jh_pg_routed_span80', 'pipelines.kimjaehak.peace_span_variants.RoutedSpan80'),
    'requested_span80': ('jh_pg_requested_span80', 'pipelines.kimjaehak.peace_span_variants.RoutedRequestedSpan80'),
    'current': ('jh_r4_material_budget', 'pipelines.kimjaehak.material_budget.Pipeline'),
    'fusion': ('jh_pg_fusion', 'pipelines.kimjaehak.peace_guard_fusion.Pipeline'),
    'peace': ('jh_parallel_guard', 'pipelines.kimjaehak.parallel_guard.Pipeline'),
    'requested': ('jh_pg_requested', 'pipelines.kimjaehak.peace_guard_variants.RequestedAction'),
    'action': ('jh_pg_action', 'pipelines.kimjaehak.peace_safety_variants.ActionSafety'),
    'context': ('jh_pg_context', 'pipelines.kimjaehak.peace_safety_variants.ContextSafety'),
    'conditioned': ('jh_pg_conditioned', 'pipelines.kimjaehak.peace_conditioned_safety.ConditionedSafety'),
    'requested_conditioned': ('jh_pg_requested_conditioned', 'pipelines.kimjaehak.peace_conditioned_safety.RequestedConditioned'),
    'relation': ('jh_pg_relation', 'pipelines.kimjaehak.peace_ground_variants.RelationGrounding'),
    'provenance': ('jh_pg_provenance', 'pipelines.kimjaehak.peace_ground_variants.ProvenanceGrounding'),
    'vote3': ('jh_pg_vote3', 'pipelines.kimjaehak.peace_voted_safety.VotedSafety'),
    'conditioned_vote3': ('jh_pg_conditioned_vote3', 'pipelines.kimjaehak.peace_voted_safety.VotedConditioned'),
    'routed': ('jh_pg_routed', 'pipelines.kimjaehak.peace_routed_safety.RoutedSafety'),
    'routed_requested': ('jh_pg_routed_requested', 'pipelines.kimjaehak.peace_routed_safety.RoutedRequested'),
    'routed_relation': ('jh_pg_routed_relation', 'pipelines.kimjaehak.peace_routed_safety.RoutedRelation'),
    'routed_provenance': ('jh_pg_routed_provenance', 'pipelines.kimjaehak.peace_routed_safety.RoutedProvenance'),
}

# A fresh interpreter imports exclusively from the selected source tree. The
# wrapper only observes original lifespan startup; it does not replace routing.
# Peace's server ignores PIPELINE, so require its actual hard-coded route too.
BOOTSTRAP = r'''
import asyncio, contextlib, functools, hashlib, inspect, json, logging, os, pathlib, sys
logging.disable(logging.CRITICAL)
try:
    root, variant, expected, port, identity_path, reference = sys.argv[1:]
    root = pathlib.Path(root).resolve()
    sys.path.insert(0, str(root))
    os.environ['PIPELINE'] = variant
    from pipelines import load
    selected = load(variant)
    if selected.__module__ + '.' + selected.__name__ != expected:
        raise RuntimeError('Selected pipeline class mismatch')
    if reference == '1' and getattr(selected, 'learned_models_enabled', None) is not False:
        raise RuntimeError('Peace reference must disable trained models before initialization')
    selected.request_slots = 64
    import server
    if pathlib.Path(server.__file__).resolve() != root / 'server.py' or server.PIPELINE != variant:
        raise RuntimeError('Actual server route mismatch')
    # Count logical method invocations, not payloads, tokens, or HTTP retries.
    # Install before original startup so any index-building calls are included.
    counts = {'instrumented': False, 'scope': 'server_lifespan_llm_method_invocations',
              'chat': {key: 0 for key in ('started', 'completed', 'cancelled', 'errors')},
              'embed': {key: 0 for key in ('started', 'completed', 'cancelled', 'errors')}}
    counts['r7_policy_calls'] = {'product':0,'other':0,'scaffold':0}
    counts['r7_guard_aggregation'] = {'requests':0,'changed':0}
    counts['r7_adaptive_ground'] = {'requests':0,'early':0,'completed_votes':0}
    counts['chat_efforts'] = {key: 0 for key in ('minimal','low','medium','high','other')}
    llm_class = getattr(server, 'LLM', None)
    def counted_method(method, counters):
        @functools.wraps(method)
        async def wrapped(self, *args, **kwargs):
            counters['started'] += 1
            if counters is counts['chat']:
                policy = sys.modules.get('pipelines.kimjaehak.round7_policy')
                system = args[0] if args else kwargs.get('system','')
                for key, attr in (('product','PRODUCT_CONTEXT_RULE'),('other','OTHER_BOUNDARY'),('scaffold','SCAFFOLD_RULE')):
                    marker = getattr(policy,attr,None)
                    if marker and isinstance(system,str) and system.endswith(marker):
                        counts['r7_policy_calls'][key] += 1
                effort = kwargs.get('effort')
                counts['chat_efforts'][effort if effort in ('minimal','low','medium','high') else 'other'] += 1
            try:
                result = await method(self, *args, **kwargs)
            except asyncio.CancelledError:
                counters['cancelled'] += 1
                raise
            except BaseException:
                counters['errors'] += 1
                raise
            else:
                counters['completed'] += 1
                return result
        return wrapped
    if llm_class is not None and all(inspect.iscoroutinefunction(getattr(llm_class, name, None))
                                     for name in ('chat', 'embed')):
        for name in ('chat', 'embed'):
            setattr(llm_class, name, counted_method(getattr(llm_class, name), counts[name]))
        counts['instrumented'] = True
    # Public-development observability only: aggregate response metadata, never text.
    counts['completion_metadata'] = {}
    if llm_class is not None:
        original_init = llm_class.__init__
        @functools.wraps(original_init)
        def measured_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            endpoint = getattr(getattr(getattr(self, '_chat', None), 'chat', None), 'completions', None)
            create = getattr(endpoint, 'create', None)
            if not callable(create):
                return
            @functools.wraps(create)
            async def measured_create(*args, **kwargs):
                response = await create(*args, **kwargs)
                effort = (kwargs.get('extra_body') or {}).get('reasoning_effort')
                effort = effort if effort in ('minimal','low','medium','high') else 'other'
                row = counts['completion_metadata'].setdefault(effort, {'responses':0,
                    'prompt_tokens':0,'completion_tokens':0,'reasoning_tokens':0,'finish_reasons':{}})
                row['responses'] += 1
                usage = getattr(response, 'usage', None)
                for key in ('prompt_tokens','completion_tokens'):
                    value = getattr(usage, key, None)
                    if isinstance(value, int): row[key] += value
                value = getattr(getattr(usage, 'completion_tokens_details', None), 'reasoning_tokens', None)
                if isinstance(value, int): row['reasoning_tokens'] += value
                for choice in getattr(response, 'choices', []) or []:
                    reason = getattr(choice, 'finish_reason', None)
                    reason = reason if reason in ('stop','length','content_filter','tool_calls','function_call') else 'other'
                    row['finish_reasons'][reason] = row['finish_reasons'].get(reason,0) + 1
                return response
            endpoint.create = measured_create
        llm_class.__init__ = measured_init
    original = server.app.router.lifespan_context
    @contextlib.asynccontextmanager
    async def observed(app):
        async with original(app):
            instance = server.state['pipeline']
            actual = type(instance)
            if actual is not selected or instance.request_slots != 64 or instance._slots._value != 64:
                raise RuntimeError('Actual pipeline instance mismatch')
            if not pathlib.Path(inspect.getfile(actual)).resolve().is_relative_to(root):
                raise RuntimeError('Pipeline imported outside selected source tree')
            if reference == '1' and getattr(instance, 'learned_models_enabled', None) is not False:
                raise RuntimeError('Peace reference must disable trained models')
            # Count explicit diagnostic booleans only; preserve the exact state
            # object and never record request/response text or predicted spans.
            original_guard = getattr(instance, 'guard', None)
            if callable(original_guard):
                counts['guard_debug'] = {key: 0 for key in (
                    'span_review_triggered', 'span_review_attempted', 'span_review_completed',
                    'span_review_success', 'span_review_changed', 'fusion_ground_skipped_fast_block',
                    'r8_failure_input', 'r8_failure_ground', 'r8_risk_safety', 'fusion_ground_cancelled_block', 'matrix_gate', 'guard_unlearned_input', 'learned_span', 'r7_hybrid_fast', 'r7_embedding_reused', 'r7_ground_early', 'r7_material_cancelled', 'r7_examples_used')}
                async def observed_guard(body):
                    result = await original_guard(body)
                    debug = getattr(result, 'debug', {})
                    for key in counts['guard_debug']:
                        counts['guard_debug'][key] += debug.get(key) is True
                    return result
                instance.guard = observed_guard
            original_tutor = getattr(instance, 'tutor', None)
            if callable(original_tutor):
                counts['tutor_debug'] = {'tutor_mlp_gate': 0, 'tutor_mlp_llm': 0, 'r7_product_check':0, 'r7_other_check':0, 'r7_scaffold_generated':0, 'r7_apply_attempted':0, 'r7_apply_completed':0, 'r7_joint_apply_started':0, 'r7_joint_apply_completed':0, 'r7_material_cancelled':0, 'r7_material_scope_used':0, 'r7_apply_abstained':0, 'r7_calculator_completed':0, 'r7_explicit_tax_scope':0, 'r7_calc_review_started':0, 'r7_calc_review_completed':0, 'r7_calc_review_changed':0, 'r7_material_verdict_used':0, 'r7_material_verdict_missing':0}
                async def observed_tutor(body):
                    result = await original_tutor(body)
                    debug = getattr(result, 'debug', {})
                    for key in counts['tutor_debug']:
                        counts['tutor_debug'][key] += debug.get(key) is True
                    return result
                instance.tutor = observed_tutor
            if variant in ('jh_pg_requested', 'jh_pg_cancel_requested') and getattr(instance, 'guard_requested_action', None) is not True:
                raise RuntimeError('Requested-action guard is not enabled')
            data = {'requested_variant': variant, 'server_variant': server.PIPELINE,
                    'class': actual.__module__ + '.' + actual.__name__, 'request_slots': 64,
                    'source_root': str(root), 'embedding_index_loaded': instance.searcher.has_embeddings,
                    'learned_models_enabled': getattr(instance, 'learned_models_enabled', None)}
            if getattr(instance,'guard_verdict_first',False) or getattr(instance,'guard_adaptive_ground',False):
                original_ground = instance._fusion_ground
                async def observed_ground(state):
                    try:
                        return await original_ground(state)
                    finally:
                        if getattr(instance,'guard_verdict_first',False):
                            counts['r7_guard_aggregation']['requests'] += 1
                            counts['r7_guard_aggregation']['changed'] += state.debug.get('r7_verdict_changed') is True
                        if getattr(instance,'guard_adaptive_ground',False):
                            counts['r7_adaptive_ground']['requests'] += 1
                            counts['r7_adaptive_ground']['early'] += state.debug.get('r7_ground_early') is True
                            votes = state.debug.get('fusion_completed_ground_votes',0)
                            if isinstance(votes,int) and not isinstance(votes,bool) and 0<=votes<=5:
                                counts['r7_adaptive_ground']['completed_votes'] += votes
                instance._fusion_ground = observed_ground
            for key in ('material_cancel_scope','tutor_material_scope','tutor_apply_missing_material_abstain','guard_adaptive_ground','tutor_apply_joint_source','tutor_apply_concrete','tutor_apply_max_chars','guard_ground_context','tutor_apply_numeric','tutor_apply_all','tutor_apply_timeout','classifier_embedding_batched','classifier_hybrid_dense_weight','classifier_hybrid_fast_threshold','classifier_representation','guard_verdict_first','r7_product_context','r7_other_boundary','r7_scaffold_guide',
                        'reasoning_guard_input_tokens','reasoning_guard_ground_tokens',
                        'reasoning_guard_input','reasoning_guard_ground','reasoning_guard_safety',
                        'reasoning_tutor_input','reasoning_tutor_source','answer_effort',
                        'tutor_mlp_threshold','tutor_mlp_single_turn','tutor_mlp_fallback',
                        'classifier_kind','classifier_threshold','classifier_seed',
                        'classifier_width','classifier_alpha','classifier_c',
                        'classifier_embedding_timeout','classifier_text_mode','classifier_common_llm_fallback',
                        'guard_allow_safety','guard_allow_safety_timeout',
                        'guard_deadline','guard_prepare_limit','guard_material_timeout',
                        'guard_ground_timeout','guard_safety_timeout','guard_requested_action',
                        'guard_safety_action_review','guard_safety_curriculum_context',
                        'guard_safety_after_input','guard_safety_launch_early','guard_short_timeout',
                        'guard_learned_input_enabled',
                        'guard_routed_safety','guard_safety_votes','guard_independent_safety',
                        'guard_ground_effort','guard_ground_max_tokens',
                        'guard_fast_input_wait','guard_vote_first',
                        'guard_gate_threshold',
                        'guard_cancel_ground_after_block',
                        'guard_span_review','guard_span_review_threshold','guard_span_review_timeout',
                        'guard_restore_span_refinement','deadline','prepare_limit','output_votes',
                        'learned_gate_threshold','learned_span_threshold'):
                data[key] = getattr(instance, key, None)
            prompts = sys.modules.get('pipelines.v2.prompts')
            block = getattr(instance, 'guard_block_system', getattr(prompts, 'BLOCK_SYSTEM', None))
            if hasattr(instance, '_block_system'):
                guard_state = instance._new_state({'messages': [
                    {'role': 'user', 'content': 'startup configuration check'}]}, True)
                block = instance._block_system(guard_state)
            safety = getattr(instance, 'guard_safety_system', None)
            if safety is None:
                method = getattr(instance, '_parallel_safety', None)
                safety = getattr(method, '__globals__', {}).get('SAFETY_SYSTEM')
            for key, text in (('guard_safety_system_sha256', safety), ('guard_block_system_sha256', block)):
                data[key] = hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None
            ground = getattr(instance, 'unsupported_system', None)
            data['unsupported_system_sha256'] = hashlib.sha256(ground.encode()).hexdigest() if isinstance(ground, str) else None
            review = getattr(instance, 'guard_span_review_system', None)
            data['guard_span_review_system_sha256'] = hashlib.sha256(review.encode()).hexdigest() if isinstance(review, str) else None
            data['guard_safety_hints_sha256'] = {k: hashlib.sha256(v.encode()).hexdigest()
                for k,v in getattr(instance, 'guard_safety_hint_systems', {}).items()}
            counts['startup_chat_efforts'] = dict(counts['chat_efforts'])
            counts['startup'] = {name: dict(counts[name]) for name in ('chat', 'embed')}
            pathlib.Path(identity_path).write_text(json.dumps(data))
            yield
    @contextlib.asynccontextmanager
    async def counted_lifespan(app):
        try:
            async with observed(app):
                yield
        finally:
            counts['total'] = {key: counts['chat'][key] + counts['embed'][key]
                               for key in ('started', 'completed', 'cancelled', 'errors')}
            try:
                pathlib.Path(identity_path).with_name('api_counts.json').write_text(json.dumps(counts))
            except OSError:
                # Observability must never replace the original lifespan error.
                pass
    server.app.router.lifespan_context = counted_lifespan
    import uvicorn
    uvicorn.run(server.app, host='127.0.0.1', port=int(port), access_log=False, log_config=None)
except BaseException as exc:
    print('SERVER_FAILURE ' + type(exc).__name__, flush=True)
    sys.exit(1)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def archive_files(data):
    """Reject links/path traversal rather than allowing archive extraction escape."""
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe snapshot member')
            if member.isdir():
                continue
            if not member.isfile() or member.name in result or name.name == '.env':
                raise ValueError('Invalid snapshot member')
            result[member.name] = archive.extractfile(member).read()
    return result


def verify_snapshot(dest, expected):
    actual = {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}
    if actual != set(expected):
        raise ValueError('Snapshot file inventory changed')
    for name, checksum in expected.items():
        path = dest / name
        if path.is_symlink() or not path.resolve().is_relative_to(dest.resolve()) or digest(path.read_bytes()) != checksum:
            raise ValueError('Snapshot file changed')


def prepare_snapshot(root, dest):
    if not dest.resolve().is_relative_to((root / 'experiments/.cache').resolve()):
        raise ValueError('Snapshot must stay in ignored experiment cache')
    commit = subprocess.check_output(['git', 'rev-parse', PEACE_COMMIT + '^{commit}'], cwd=root, text=True).strip()
    if commit != PEACE_COMMIT:
        raise ValueError('Unexpected reference commit')
    archive = subprocess.check_output(['git', 'archive', '--format=tar', commit], cwd=root)
    files = archive_files(archive)
    expected = {name: digest(data) for name, data in sorted(files.items())}
    if not dest.exists():
        dest.mkdir(parents=True)
        for name, data in files.items():
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    verify_snapshot(dest, expected)
    return {'commit': commit, 'archive_sha256': digest(archive), 'files': expected}


def protected_hashes(root):
    paths = [root / 'SPEC.md', root / 'server.py', root / 'harness/score_local.py']
    for folder in ('core', 'grader', 'kb', 'public'):
        paths += [p for p in (root / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted(paths)}


def pipeline_hashes(root):
    return {str(p.relative_to(root)): digest(p.read_bytes())
            for p in sorted((root / 'pipelines').rglob('*.py'))}


def verify_common_inputs(root, reference, snapshot):
    """The reference may have a different server, never different KB/gold/grader."""
    current = protected_hashes(root)
    for name, checksum in current.items():
        if name != 'server.py' and snapshot['files'].get(name) != checksum:
            raise ValueError('Reference core/evaluator/KB/public differs: ' + name)
    return current


def validate_identity(value, label, source_root):
    variant, expected = CANDIDATES[label]
    if (value.get('requested_variant') != variant or value.get('server_variant') != variant
            or value.get('class') != expected or value.get('request_slots') != 64
            or value.get('source_root') != str(source_root.resolve())
            or (label == 'peace' and value.get('learned_models_enabled') is not False)
            or (label in ('requested', 'cancel_requested') and value.get('guard_requested_action') is not True)):
        raise ValueError('Recorded startup identity does not match requested candidate')


def server_env(env, root, embedding_mode):
    result = dict(env)
    result.pop('PYTHONPATH', None)
    result['PYTHONDONTWRITEBYTECODE'] = '1'
    result['KB_DIR'] = str(root / 'kb')
    result['KB_INDEX'] = str(root / 'index/kb_embeddings.json')
    if embedding_mode == 'off':
        for key in ('EMBEDDING_BASE_URL', 'EMBEDDING_MODEL', 'EMBEDDING_API_KEY'):
            result[key] = ''
    return result


def collect_candidate(label, source_root, items, dest, env, judge_workers, embedding_mode, dataset_metadata=None):
    """Small process adapter; requests and official scoring are shared, not copied."""
    dest.mkdir()
    variant, expected = CANDIDATES[label]
    identity_path = dest / 'startup_identity.json'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    manifest = {'variant': variant, 'candidate': label, 'expected_class': expected,
                'workers': 64, 'request_slots_override': 64, 'judge_workers': judge_workers,
                'count': len(items), 'endpoints': ['guard'], 'score_scope': 'guard',
                'evaluator_hashes': shared.evaluator_hashes(), 'embedding_mode': embedding_mode,
                'scoring_deferred': True, 'judge_enabled': True}
    manifest['endpoints'] = sorted({item['endpoint'] for item in items})
    manifest['score_scope'] = '+'.join(manifest['endpoints'])
    if dataset_metadata is not None:
        manifest['dataset_provenance'] = dataset_metadata
        manifest['scope'] = dataset_metadata['scope']
    write_json(dest / 'manifest.json', manifest)
    command = [sys.executable, '-B', '-c', BOOTSTRAP, str(source_root), variant, expected,
               str(port), str(identity_path), '1' if label == 'peace' else '0']
    with (dest / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=source_root, env=env, stdout=log, stderr=log)
        try:
            shared.wait_ready(proc, port)
            identity = json.loads(identity_path.read_text())
            validate_identity(identity, label, source_root)
            manifest['effective_config'] = identity
            write_json(dest / 'manifest.json', manifest)
            started = time.monotonic()
            rows = asyncio.run(shared.collect(items, port, env['TEAM_TOKEN'], 64, dest / 'responses.jsonl'))
            seconds = round(time.monotonic() - started, 3)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    durations = sorted(row['seconds'] for row in rows)
    errors = re.findall(r'errors=([^\s]+)', (dest / 'server.log').read_text())
    runtime = {'collect_seconds': seconds, 'p50': statistics.median(durations),
               'p95': durations[min(len(durations)-1, int(.95*len(durations)))], 'max': max(durations),
               'http_errors': sum(bool(row['error']) for row in rows),
               'stage_failure_requests': sum(error != '-' for error in errors),
               'stage_errors': dict(Counter(part for error in errors if error != '-'
                                            for part in error.split(',')))}
    # Read only after SIGTERM/wait: graceful lifespan shutdown writes final counts.
    counts_path = dest / 'api_counts.json'
    runtime['api_calls'] = json.loads(counts_path.read_text()) if counts_path.exists() else {
        'instrumented': False, 'available': False, 'reason': 'shutdown_counts_unavailable'}
    write_json(dest / 'runtime.json', runtime)
    if shared.evaluator_hashes() != manifest['evaluator_hashes']:
        raise RuntimeError('Official evaluator changed during collection')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataset', choices=datasets.DATASETS, default='public')
    parser.add_argument('--order', nargs='+', choices=CANDIDATES, default=['current','fusion','peace'])
    parser.add_argument('--embedding-mode', choices=('env', 'off'), default='env')
    parser.add_argument('--judge-workers', type=int, default=32)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--prepare-only', action='store_true')
    phase.add_argument('--collect-only', action='store_true')
    phase.add_argument('--score-only', action='store_true')
    args = parser.parse_args(argv)
    if (not re.fullmatch(r'[A-Za-z0-9_-]+', args.name) or len(set(args.order)) != len(args.order)
            or not 1 <= args.judge_workers <= 64):
        parser.error('Invalid name, order or judge workers')
    dest = ROOT / 'experiments/.cache' / ('jh_guard_fusion_' + args.name)
    if not args.score_only:
        dest.mkdir(exist_ok=False)
        snapshot = prepare_snapshot(ROOT, dest / 'peace_snapshot')
        protected = verify_common_inputs(ROOT, dest / 'peace_snapshot', snapshot)
        items, dataset_metadata = datasets.load_dataset(args.dataset, ROOT)
        data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in items)
        (dest / 'inputs.jsonl').write_text(data)
        plan = {'scope': dataset_metadata['scope'], 'dataset_provenance': dataset_metadata, 'order': args.order,
                'peace_snapshot': snapshot, 'protected_hashes': protected,
                'pipeline_hashes': pipeline_hashes(ROOT), 'bootstrap_sha256': digest(BOOTSTRAP.encode()),
                'runner_sha256': digest(Path(__file__).read_bytes()),
                'dataset_module_sha256': digest(Path(datasets.__file__).read_bytes()),
                'dataset_sha256': digest(data.encode()), 'embedding_mode': args.embedding_mode,
                'current_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'environment_policy': 'One inherited environment for all children; no secrets recorded'}
        write_json(dest / 'plan.json', plan)
    else:
        plan = json.loads((dest / 'plan.json').read_text())
        if plan['order'] != args.order or plan['embedding_mode'] != args.embedding_mode:
            raise ValueError('Resume settings differ from frozen plan')
        data = (dest / 'inputs.jsonl').read_bytes()
        if digest(data) != plan['dataset_sha256']:
            raise ValueError('Frozen public dataset changed')
        items = [json.loads(line) for line in data.splitlines()]
        expected_items, dataset_metadata = datasets.load_dataset(args.dataset, ROOT)
        if (items != expected_items
                or plan.get('dataset_provenance', {}).get('dataset', 'public') != args.dataset):
            raise ValueError('Resume dataset differs from pinned inputs')
    current_protected = protected_hashes(ROOT)
    if args.score_only:
        # Cached responses no longer execute pipeline/server/recorder code.
        # Retain their recorded code hashes, but permit newer experimental code.
        scoring_files = lambda rows: {k: v for k, v in rows.items()
                                     if k.startswith(('grader/', 'harness/', 'kb/', 'public/'))}
        protected_equal = scoring_files(current_protected) == scoring_files(plan['protected_hashes'])
    else:
        protected_equal = current_protected == plan['protected_hashes']
    if not protected_equal:
        raise ValueError('Protected current files changed')
    # Older public plans did not list SPEC separately; their exact reference
    # snapshot still pins its original content for resumed scoring.
    expected_spec = plan['protected_hashes'].get('SPEC.md', plan['peace_snapshot']['files'].get('SPEC.md'))
    if expected_spec is not None and digest((ROOT / 'SPEC.md').read_bytes()) != expected_spec:
        raise ValueError('Original competition specification changed')
    if not args.score_only and (pipeline_hashes(ROOT) != plan['pipeline_hashes']
            or digest(BOOTSTRAP.encode()) != plan['bootstrap_sha256']
            or digest(Path(__file__).read_bytes()) != plan['runner_sha256']
            or digest(Path(datasets.__file__).read_bytes()) != plan['dataset_module_sha256']):
        raise ValueError('Experiment implementation changed after preparation')
    verify_snapshot(dest / 'peace_snapshot', plan['peace_snapshot']['files'])
    if args.prepare_only:
        print(json.dumps({'prepared': str(dest), 'count': len(items), 'api_calls': 0}))
        return
    from experiments.jh_environment import load_environment
    load_environment(ROOT)
    for key in ('TEAM_TOKEN', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
        if not os.environ.get(key):
            raise ValueError('Required environment setting missing: ' + key)
    env = server_env(os.environ, ROOT, args.embedding_mode)
    if not args.score_only:
        for label in args.order:
            source = dest / 'peace_snapshot' if label == 'peace' else ROOT
            collect_candidate(label, source, items, dest / label, env, args.judge_workers,
                              args.embedding_mode, dataset_metadata)
            verify_snapshot(dest / 'peace_snapshot', plan['peace_snapshot']['files'])
            if protected_hashes(ROOT) != plan['protected_hashes']:
                raise ValueError('Protected files changed during collection')
            if pipeline_hashes(ROOT) != plan['pipeline_hashes']:
                raise ValueError('Candidate implementation changed during collection')
    if args.collect_only:
        print('SCORING_PENDING', dest)
        return
    summary = []
    for label in args.order:
        run = dest / label
        manifest = json.loads((run / 'manifest.json').read_text())
        source = dest / 'peace_snapshot' if label == 'peace' else ROOT
        validate_identity(manifest['effective_config'], label, source)
        if manifest['evaluator_hashes'] != shared.evaluator_hashes():
            raise ValueError('Scorer differs from collection')
        rows = [json.loads(line) for line in (run / 'responses.jsonl').read_text().splitlines()]
        result = shared.score_rows(items, rows, args.judge_workers, False, run / 'judge.jsonl')
        if args.dataset == 'counterfactual':
            from experiments.jh_guard_counterfactual import grounding_metrics
            result['counterfactual_details'] = grounding_metrics(items, rows)
        elif args.dataset == 'relational':
            result['relational_details'] = datasets.relational_metrics(items, rows)
        elif args.dataset == 'intent':
            result['intent_details'] = datasets.intent_metrics(items, rows)
        if shared.evaluator_hashes() != manifest['evaluator_hashes']:
            raise ValueError('Official evaluator changed during scoring')
        from experiments.jh_neural_peace_report import safety_metrics
        result['safety_details'] = safety_metrics(items, rows)
        result['score_scope'] = 'guard'
        result['diagnostic_scope'] = dataset_metadata['scope']
        result['dataset_provenance'] = dataset_metadata
        result['runtime'] = json.loads((run / 'runtime.json').read_text())
        write_json(run / 'result.json', result)
        summary.append(datasets.summarize_result(label, result, dataset_metadata))
    write_json(dest / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('COMPARISON_FAILURE ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
