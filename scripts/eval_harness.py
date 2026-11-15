"""Diagnostic Evaluation Harness for /tutor and /guard.

Measures component-level metrics:
- Retrieval Recall@1, Recall@2, Recall@4 on public/tutor.jsonl
- Citation Precision, Recall, F1, and Multi-Citation Penalty
- 7-Action Accuracy & Confusion Summary + Over-refusal / Under-refusal (Compliance) counts
- Guard Binary Block F1 (TP, FP, TN, FN, Precision, Recall)
- Guard Non-Normal Category Accuracy (9 classes)
- Guard Output Grounded Accuracy, Safe Accuracy, and Verbatim Unsupported Span F1

Usage:
    # Fast in-process deterministic evaluation (no Judge LLM needed):
    python scripts/eval_harness.py --in-process --no-judge

    # Evaluate against a running server:
    python scripts/eval_harness.py --tutor-url http://localhost:8000/tutor --guard-url http://localhost:8000/guard --token dev-token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "grader") not in sys.path:
    sys.path.insert(0, str(ROOT / "grader"))

import scoring
from judge import make_judge
from harness.score_local import as_scoring_input, load_kb_dir


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def _measure_retrieval_recall(kb_index: Any, tutor_items: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute Retrieval Recall@1, Recall@2, Recall@4 on tutor items that have gold evidence."""
    from app.tutor_pipeline import TutorPipeline, _strip_for_canned_check, _CANNED_TURN0_NORM
    from app.normalize import collapse_korean_obfuscation

    hits_at = {1: 0, 2: 0, 4: 0}
    total_with_ev = 0

    for item in tutor_items:
        ev = set(item.get("labels", {}).get("evidence") or [])
        if not ev:
            continue
        total_with_ev += 1
        messages = item.get("messages") or []
        user_turns = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str) and m.get("content", "").strip()
        ]
        if not user_turns:
            continue
        last_q = collapse_korean_obfuscation(user_turns[-1])
        prev_q = None
        if len(user_turns) > 1:
            cands = user_turns[:-1]
            if _strip_for_canned_check(cands[0]) in _CANNED_TURN0_NORM:
                cands = cands[1:]
            if cands:
                prev_q = collapse_korean_obfuscation(" ".join(cands[-2:]))

        secs = await kb_index.retrieve(last_q, top_k_sections=4, prev_query=prev_q, last_weight=0.75)
        sids = [s.section_id for s in secs]
        for k in (1, 2, 4):
            if ev & set(sids[:k]):
                hits_at[k] += 1

    if total_with_ev == 0:
        return {"recall@1": 0.0, "recall@2": 0.0, "recall@4": 0.0, "count": 0}
    return {
        "recall@1": round(100.0 * hits_at[1] / total_with_ev, 2),
        "recall@2": round(100.0 * hits_at[2] / total_with_ev, 2),
        "recall@4": round(100.0 * hits_at[4] / total_with_ev, 2),
        "count": total_with_ev,
    }


def _build_in_process_caller(mode: str) -> tuple[Callable[[str, Dict[str, Any]], Dict[str, Any]], Any]:
    os.environ["PIPELINE_MODE"] = mode
    from app.config import ClientPool, load_settings
    from app.guard_classifier import GuardClassifier
    from app.guard_verifier import GuardVerifier
    from app.kb_index import KBIndex
    from app.tutor_pipeline import TutorPipeline

    settings = load_settings()
    pool = ClientPool(settings)
    kb_index = KBIndex(settings, embed_client=pool.embed_client)
    guard_clf = GuardClassifier(settings, pool)
    guard_clf.load_or_train()
    guard_ver = GuardVerifier(settings, pool, kb_index)

    if mode == "supervisor":
        try:
            from app.supervisor_graph import SupervisorTutorPipeline
            tutor_pipe: Any = SupervisorTutorPipeline(
                settings, pool, kb_index, guard_classifier=guard_clf, guard_verifier=guard_ver
            )
        except ImportError:
            tutor_pipe = TutorPipeline(settings, pool, kb_index, guard_classifier=guard_clf)
    else:
        tutor_pipe = TutorPipeline(settings, pool, kb_index, guard_classifier=guard_clf)

    loop = asyncio.new_event_loop()
    if kb_index is not None and pool.embed_client is not None:
        try:
            loop.run_until_complete(kb_index.ensure_embeddings())
        except Exception:
            pass

    def call(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if endpoint == "tutor":
            return loop.run_until_complete(tutor_pipe.handle(payload["messages"]))
        decision, category, _ = loop.run_until_complete(guard_clf.classify(payload["messages"]))
        res: Dict[str, Any] = {"input": {"decision": decision, "category": category}}
        if payload.get("candidate_answer") is not None:
            res["output"] = loop.run_until_complete(
                guard_ver.verify(payload["messages"], payload["candidate_answer"], decision, category)
            )
        return res

    return call, kb_index


def run_diagnostics(
    public_items: List[Dict[str, Any]],
    call: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    judge: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]],
    kb: Dict[str, str],
    workers: int = 8,
) -> Dict[str, Any]:
    def _one(p: Dict[str, Any]):
        item, v = as_scoring_input(p)
        payload: Dict[str, Any] = {
            "spec_version": "1.0",
            "session_id": "s_" + secrets.token_hex(8),
            "messages": v["messages"],
        }
        if v["candidate_answer"]:
            payload["candidate_answer"] = v["candidate_answer"]
        t0 = time.perf_counter()
        call_err: Optional[str] = None
        try:
            resp = call(p["endpoint"], payload)
        except Exception as exc:
            resp = None
            call_err = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if p["endpoint"] == "tutor":
            r = scoring.score_tutor(item, v, resp, judge or (lambda k, pl: (_ for _ in ()).throw(RuntimeError())), kb)
            if judge is None:
                r["judge_error"] = False
            pts = scoring.tutor_points(r)
        else:
            r = scoring.score_guard(item, v, resp)
            pts = scoring.guard_points(r)
        return p, r, resp, pts, elapsed_ms, call_err

    with ThreadPoolExecutor(max_workers=workers) as ex:
        raw_results = list(ex.map(_one, public_items))

    call_errors = Counter(err for _, _, _, _, _, err in raw_results if err)
    tutor_rows = [(p, r, resp, pts, ms) for p, r, resp, pts, ms, _ in raw_results if p["endpoint"] == "tutor"]
    guard_rows = [(p, r, resp, pts, ms) for p, r, resp, pts, ms, _ in raw_results if p["endpoint"] == "guard"]

    agg = scoring.aggregate([r for _, r, _, _, _ in tutor_rows], [r for _, r, _, _, _ in guard_rows], {"tutor": 0.5, "guard": 0.5})

    # Tutor Detailed Diagnostics
    action_tp = sum(1 for _, r, _, _, _ in tutor_rows if r["action_correct"])
    over_refusals = sum(1 for _, r, _, _, _ in tutor_rows if r.get("over_refusal"))
    complied_cnt = sum(1 for _, r, _, _, _ in tutor_rows if r.get("complied") is True)
    cit_f1_list: List[float] = []
    cit_p_list: List[float] = []
    cit_r_list: List[float] = []
    multi_cite_cnt = 0
    action_conf: Dict[str, Counter] = defaultdict(Counter)

    for p, r, resp, _, _ in tutor_rows:
        gold_act = p["labels"]["action"]
        pred_act = resp.get("action", "INVALID") if isinstance(resp, dict) else "INVALID"
        action_conf[gold_act][pred_act] += 1
        if isinstance(resp, dict) and len(resp.get("citations") or []) > 1:
            multi_cite_cnt += 1
        if r["citation_p"] is not None:
            cp, cr = r["citation_p"], r["citation_r"]
            cit_p_list.append(cp)
            cit_r_list.append(cr)
            cit_f1_list.append(2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0)

    # Guard Detailed Diagnostics
    g_records = [r for _, r, _, _, _ in guard_rows]
    tp = sum(1 for r in g_records if r["gold_block"] and r["input_correct"])
    fn = sum(1 for r in g_records if r["gold_block"] and not r["input_correct"])
    fp = sum(1 for r in g_records if not r["gold_block"] and not r["input_correct"])
    tn = sum(1 for r in g_records if not r["gold_block"] and r["input_correct"])
    blk_f1 = scoring.block_f1(g_records)

    cat_scored = [r["category_correct"] for r in g_records if r["category_scored"]]
    out_records = [r for r in g_records if r["has_output"] and r["valid"]]
    grounded_acc = (
        sum(1 for r in out_records if r["output_correct"]) / len(out_records)
        if out_records
        else 0.0
    )
    safe_acc = (
        sum(1 for r in out_records if r["safe_correct"]) / len(out_records)
        if out_records
        else 0.0
    )
    avg_span_f1 = (
        sum(r["span_f1"] for r in out_records) / len(out_records)
        if out_records
        else 0.0
    )

    return {
        "aggregate": agg,
        "call_errors": dict(call_errors.most_common(5)),
        "tutor_diagnostics": {
            "count": len(tutor_rows),
            "action_accuracy": round(100.0 * action_tp / max(1, len(tutor_rows)), 2),
            "citation_precision": round(100.0 * sum(cit_p_list) / max(1, len(cit_p_list)), 2),
            "citation_recall": round(100.0 * sum(cit_r_list) / max(1, len(cit_r_list)), 2),
            "citation_f1": round(100.0 * sum(cit_f1_list) / max(1, len(cit_f1_list)), 2),
            "multi_citation_count": multi_cite_cnt,
            "over_refusal_count": over_refusals,
            "complied_attack_count": complied_cnt,
            "action_confusion": {k: dict(v) for k, v in sorted(action_conf.items())},
        },
        "guard_diagnostics": {
            "count": len(guard_rows),
            "block_f1": round(100.0 * blk_f1, 2),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "non_normal_category_accuracy": round(100.0 * sum(cat_scored) / max(1, len(cat_scored)), 2),
            "output_grounded_accuracy": round(100.0 * grounded_acc, 2),
            "output_safe_accuracy": round(100.0 * safe_acc, 2),
            "output_span_f1": round(100.0 * avg_span_f1, 2),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnostic Evaluation Harness")
    ap.add_argument("--in-process", action="store_true", help="Run pipeline directly in-process without HTTP server.")
    ap.add_argument("--mode", default=os.environ.get("PIPELINE_MODE", "single"), choices=["single", "supervisor"])
    ap.add_argument("--tutor-url", default="http://localhost:8000/tutor")
    ap.add_argument("--guard-url", default="http://localhost:8000/guard")
    ap.add_argument("--token", default=os.environ.get("TEAM_TOKEN", "dev-token"))
    ap.add_argument("--public", default=str(ROOT / "public"))
    ap.add_argument("--kb", default=str(ROOT / "kb"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true", help="Skip LLM judge calls for fast deterministic metrics.")
    ap.add_argument("--output-json", type=Path, default=None, help="Optional path to save diagnostic JSON report.")
    args = ap.parse_args(argv)

    pub_dir = Path(args.public)
    tutor_items = _read_jsonl(pub_dir / "tutor.jsonl")
    guard_items = _read_jsonl(pub_dir / "guard.jsonl")
    all_items = tutor_items + guard_items

    kb_index = None
    if args.in_process:
        call, kb_index = _build_in_process_caller(args.mode)
        workers = 1  # in-process event loop is single-threaded
    else:
        urls = {"tutor": args.tutor_url, "guard": args.guard_url}
        opener = urllib.request.build_opener()

        def call(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            req = urllib.request.Request(
                urls[endpoint],
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {args.token}",
                },
            )
            with opener.open(req, timeout=30.0) as resp:
                return json.loads(resp.read(256 * 1024 + 1))

        workers = args.workers

    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    judge = make_judge(**env) if (not args.no_judge and all(env.values())) else None

    report = run_diagnostics(all_items, call, judge, load_kb_dir(args.kb), workers=workers)

    if kb_index is not None:
        ret_metrics = asyncio.run(_measure_retrieval_recall(kb_index, tutor_items))
        report["retrieval_diagnostics"] = ret_metrics

    print("=" * 78)
    target_desc = f"in-process:{args.mode}" if args.in_process else "http"
    print(f"DIAGNOSTIC HARNESS REPORT (target={target_desc}, judge={'ON' if judge else 'OFF'})")
    print("=" * 78)
    if "retrieval_diagnostics" in report:
        rd = report["retrieval_diagnostics"]
        print(f"[Retrieval] Recall@1={rd['recall@1']}% | Recall@2={rd['recall@2']}% | Recall@4={rd['recall@4']}% (n={rd['count']})")
    td = report["tutor_diagnostics"]
    print(
        f"[Tutor]     Score={report['aggregate']['tutor']} | ActionAcc={td['action_accuracy']}% | "
        f"Cit_F1={td['citation_f1']}% (P={td['citation_precision']}%, R={td['citation_recall']}%) | "
        f"MultiCite={td['multi_citation_count']} | OverRefuse={td['over_refusal_count']}"
    )
    gd = report["guard_diagnostics"]
    print(
        f"[Guard]     Score={report['aggregate']['guard']} | Block_F1={gd['block_f1']}% "
        f"(TP={gd['tp']}, FP={gd['fp']}, FN={gd['fn']}, TN={gd['tn']}) | "
        f"CatAcc={gd['non_normal_category_accuracy']}% | GroundedAcc={gd['output_grounded_accuracy']}% | "
        f"SafeAcc={gd['output_safe_accuracy']}% | SpanF1={gd['output_span_f1']}%"
    )
    print(
        f"[Total]     {report['aggregate']['total']} (invalid={report['aggregate']['invalid']}, "
        f"judge_errors={report['aggregate']['judge_errors']})"
    )
    if report.get("call_errors"):
        print(f"[CallErrs]  {report['call_errors']}")
    print("=" * 78)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
