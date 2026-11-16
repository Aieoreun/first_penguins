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
