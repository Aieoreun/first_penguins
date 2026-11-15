"""Export aggregate public experiment results; omit prompts, responses and secrets."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root=ROOT):
    records = []
    hashes = None
    for path in sorted((root / "experiments/.cache").glob("jh_*/**/result.json")):
        run = path.parent
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            continue
        m = json.loads(manifest_path.read_text())
        r = json.loads(path.read_text())
        count = m.get("count", len(r.get("items", [])))
        if count < 200:
            continue  # Small smoke tests are not quality estimates.
        if hashes is not None and hashes != m["evaluator_hashes"]:
            raise ValueError("Cannot combine results from different evaluators")
        hashes = m["evaluator_hashes"]
        component = m.get("scope") == "public_output_component_only"
        scope = m.get("scope", r.get("score_scope"))
        transport = m.get("http_transport", "legacy_default_thread_pool")
        requested = m["workers"]
        # Historical runs were made on this host, where the pool had 16 threads.
        effective = None if component else requested if transport == "explicit_worker_thread_pool" else min(16, requested)
        trained = (m.get("effective_config") or {}).get("training_source") == "bundled_public_categories"
        records.append({
            "run": str(run.relative_to(root)), "variant": m["variant"], "commit": m["commit"],
            "started_at": m.get("started_at"), "scoring_deferred": m.get("scoring_deferred", False),
            "validation_regime": "public_in_sample_learned_classifier" if trained else "public_development",
            "scope": scope, "count": count,
            "workers": requested, "effective_http_workers": effective,
            "http_transport": None if component else transport,
            "request_slots_override": m.get("request_slots_override"),
            "judge_enabled": m.get("judge_enabled", False),
            "quality_comparison_valid": run.parent.name != "jh_guided_input_guard" and
                                        (component or scope == "guard" or m.get("judge_enabled", False)),
            "result_sha256": digest(path), "responses_sha256": digest(run / "responses.jsonl"),
            "scores": {key: r[key] for key in ("total", "tutor", "guard", "invalid", "judge_errors")},
            "guard_details": {key: value for key, value in r["guard_details"].items() if key != "confusion"},
            "per_category": r["per_category"], "runtime": r.get("runtime"),
            **({"source": m["source"]} if component else {}),
        })
    return {"updated": datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
            "official_hidden_evaluation": False, "evaluator_hashes": hashes, "runs": records}


if __name__ == "__main__":
    output = ROOT / "docs/jh/results.json"
    data = summarize()
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(data['runs'])} completed public runs to {output}")
