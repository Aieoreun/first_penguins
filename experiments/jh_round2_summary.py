"""Summarize second-round local diagnostics without additional model calls."""
import hashlib
import json
from pathlib import Path

from jh_compare import ROOT, evaluator_hashes
from jh_tutor_components import explain


def summarize():
    records, pending, failed = [], [], []
    cache = ROOT / "experiments/.cache"
    for manifest_path in sorted(cache.glob("*r2*/*/manifest.json")):
        run = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        if "variant" not in manifest:
            continue
        if (run / "failure.json").exists():
            failed.append({"run":str(run.relative_to(ROOT)),
                           **json.loads((run / "failure.json").read_text())})
            continue
        if not (run / "runtime.json").exists():
            pending.append({"run":str(run.relative_to(ROOT)),"phase":"collection"})
            continue
        if manifest["evaluator_hashes"] != evaluator_hashes():
            raise ValueError("Evaluator changed")
        completed = next((run / name for name in ("result.json", "derived_result.json", "diagnostic.json")
                          if (run / name).exists()), None)
        if completed is None:
            pending.append({"run":str(run.relative_to(ROOT)),"phase":"scoring"})
            continue
        result = json.loads(completed.read_text())
        record = {"run": str(run.relative_to(ROOT)),
                  "variant": manifest["variant"], "count": manifest["count"], "commit": manifest["commit"],
                  "dirty": manifest["dirty"], "embedding_mode": manifest["embedding_mode"],
                  "embedding_index_loaded": manifest.get("embedding_index_loaded"),
                  "judge_workers": manifest["judge_workers"],
                  "scope": result.get("scope", "bundled_public_in_sample"),
                  "score_scope": result.get("score_scope", "+".join(manifest.get("endpoints", []))),
                  "result_sha256": hashlib.sha256(completed.read_bytes()).hexdigest(),
                  "runtime": json.loads((run / "runtime.json").read_text())}
        if record["scope"] == "public_paraphrase_input_diagnostic":
            record.update(input_metrics=result["input_metrics"],invalid=result["invalid"])
        else:
            fields = ("tutor", "invalid", "judge_errors") if record["scope"] == "provided_kb_fact_diagnostic" else (
                "total", "tutor", "guard", "guard_details", "invalid", "judge_errors")
            record.update({k:result[k] for k in fields})
            components = run / "tutor_components.json"
            if not components.exists():
                components.write_text(json.dumps(explain(run),indent=2))
            component_result = json.loads(components.read_text())
            record["tutor_components"] = component_result["verified_summary"]
            record["unverified_components"] = component_result["unverified_items"]
        records.append(record)
    return {"scope":"round2_development_summary_not_private_evaluation", "records":records,"pending":pending,"failed":failed,
            "official_user_report":{"total":82.48,"tutor":76.75,"guard":88.21,"duration":57.039,
                                    "count":600,"seed":"24eecaf226f952687cf1f8dfea58b696",
                                    "runbox_commit_independently_verified":False}}


if __name__ == "__main__":
    result = summarize()
    target = ROOT / "docs/jh/round2_results.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"completed":len(result["records"]),"pending":result["pending"]}))
