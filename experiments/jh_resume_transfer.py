"""Resume original scoring for a completed PUBLIC-DERIVED full diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from jh_compare import ROOT, evaluator_hashes, score_rows


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env",override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "experiments/.cache") or (run / "derived_result.json").exists():
        parser.error("Use an unscored local derived run")
    manifest = json.loads((run / "manifest.json").read_text())
    provenance = json.loads((run.parent / "derivation.json").read_text())
    if manifest.get("scope") not in {"public_paraphrase_full_diagnostic", "provided_kb_fact_diagnostic"} or provenance["scope"] != manifest["scope"]:
        parser.error("This is not a full public-derived diagnostic")
    if manifest["model"] != os.environ["OPENAI_MODEL"] or manifest["evaluator_hashes"] != evaluator_hashes():
        parser.error("Model or original evaluator differs")
    dataset = run.parent / "derived_inputs.jsonl"
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != provenance["dataset_sha256"]:
        parser.error("Derived input data changed")
    items = list(map(json.loads,dataset.read_text().splitlines()))
    rows = list(map(json.loads,(run / "responses.jsonl").read_text().splitlines()))
    if len(items) != manifest["count"]:
        parser.error("Derived item count differs")
    result = score_rows(items,rows,manifest["judge_workers"],journal=run / "judge.jsonl")
    result.update(provenance)
    result["runtime"] = json.loads((run / "runtime.json").read_text())
    (run / "derived_result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({key:result[key] for key in ("scope","count","total","tutor","guard","judge_errors","invalid")}),flush=True)


if __name__ == "__main__":
    main()
