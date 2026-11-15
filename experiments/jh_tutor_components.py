"""Explain completed tutor scores using cached judge decisions only; no API calls.

Concurrent duplicate judge calls can disagree. Such items and any mismatch with
the immutable original item score are excluded from verified component averages.
"""
import argparse
import hashlib
import json
from pathlib import Path

from jh_compare import ROOT, as_scoring_input, evaluator_hashes, load_kb_dir, load_public, scoring


def explain(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["evaluator_hashes"] != evaluator_hashes():
        raise ValueError("Original evaluator changed")
    derived = (run / "derived_result.json").exists()
    result_path = run / ("derived_result.json" if derived else "result.json")
    result = json.loads(result_path.read_text())
    if derived:
        dataset = run.parent / "derived_inputs.jsonl"
        provenance = json.loads((run.parent / "derivation.json").read_text())
        if hashlib.sha256(dataset.read_bytes()).hexdigest() != provenance["dataset_sha256"]:
            raise ValueError("Derived dataset changed")
        items = list(map(json.loads, dataset.read_text().splitlines()))
    else:
        items = load_public()
    responses = {r["id"]: r["response"] for r in map(json.loads, (run / "responses.jsonl").read_text().splitlines())}
    points = {r["id"]: r["points"] for r in result["items"]}
    decisions, ambiguous = {}, set()
    for entry in map(json.loads, (run / "judge.jsonl").read_text().splitlines()):
        key, value = entry["key"], entry["result"]
        if key in decisions and decisions[key] != value:
            ambiguous.add(key)
        decisions[key] = value
    kb = load_kb_dir(ROOT / "kb")
    details = []
    for public in items:
        if public["endpoint"] != "tutor" or public["id"] not in points:
            continue
        used = []
        def recorded_judge(kind, payload):
            key = hashlib.sha256(json.dumps([kind, payload], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            used.append(key)
            if key not in decisions:
                raise scoring.JudgeError("No recorded decision")
            return decisions[key]
        item, variant = as_scoring_input(public)
        r = scoring.score_tutor(item, variant, responses[public["id"]], recorded_judge, kb)
        recomputed = round(100 * scoring.tutor_points(r), 1)
        consistent = recomputed == points[public["id"]]
        verified = consistent and not r["judge_error"] and not any(key in ambiguous for key in used)
        if r["citation_p"] is not None:
            p, recall = r["citation_p"], r["citation_r"]
            r["citation_f1"] = 2 * p * recall / (p + recall) if p + recall else 0.
        else:
            r["citation_f1"] = None
        details.append({"id": public["id"], "category": public["labels"]["category"],
                        "verified": verified, "matches_original_points": consistent,
                        "original_points": points[public["id"]], **r})
    def averages(rows):
        out = {"items": len(rows)}
        for field in ("action_correct", "citation_f1", "facts", "faithful", "complied"):
            values = [r[field] for r in rows if r.get(field) is not None]
            out[field] = {"count": len(values), "mean": sum(values) / len(values) if values else None}
        return out
    valid = [row for row in details if row["verified"]]
    return {"scope": "cached_tutor_component_diagnostic", "source": str(run.relative_to(ROOT)),
            "source_scope": result.get("scope", result.get("score_scope")),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "no_new_judge_calls": True, "ambiguous_judge_keys": len(ambiguous),
            "unverified_items": len(details) - len(valid), "verified_summary": averages(valid),
            "verified_categories": {category: averages([r for r in valid if r["category"] == category])
                                    for category in sorted({r["category"] for r in valid})},
            "items": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "experiments/.cache") or (run / "tutor_components.json").exists():
        parser.error("Use a completed local run without existing component diagnostics")
    report = explain(run)
    (run / "tutor_components.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("source", "ambiguous_judge_keys", "unverified_items", "verified_summary")}), flush=True)
