"""PUBLIC-only fixed-policy wording diagnostic with our input predictions frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from jh_compare import ROOT, evaluator_hashes, load_public, score_rows
from core.schema import POLICY
from pipelines.kimjaehak.calibrated import FIXED
from pipelines.kimjaehak.brief_policy import POLICY_TEXT


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--style", choices=("original", "brief"), required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache") or not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("Invalid source or name")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["evaluator_hashes"] != evaluator_hashes() or manifest.get("scope"):
        parser.error("Use an original-public run with the current evaluator")
    originals = {FIXED.get(label, POLICY[label].fixed_answer): label for label in POLICY_TEXT}
    prior = {row["id"]: row for row in map(json.loads, (source / "responses.jsonl").read_text().splitlines())}
    items, responses = [], []
    for item in load_public(endpoints=("tutor",)):
        old = prior[item["id"]]
        response = old.get("response") or {}
        label = originals.get(response.get("answer"))
        if label is None:
            continue
        # Select using our emitted fixed response, never the gold category.
        changed = dict(response)
        if args.style == "brief":
            changed["answer"] = POLICY_TEXT[label]
        items.append(item)
        responses.append({**old, "response": changed})
    if not items:
        parser.error("No recognized fixed-policy responses")
    dest = ROOT / "experiments/.cache" / ("jh_policy_" + args.name)
    dest.mkdir(exist_ok=False)
    provenance = {"scope": "public_fixed_policy_component_only", "count": len(items), "style": args.style,
                  "source": str(source.relative_to(ROOT)), "source_manifest": manifest,
                  "source_sha256": hashlib.sha256((source / "responses.jsonl").read_bytes()).hexdigest(),
                  "evaluator_hashes": evaluator_hashes(),
                  "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (dest / "manifest.json").write_text(json.dumps(provenance, indent=2))
    (dest / "responses.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in responses))
    result = score_rows(items, responses, 8, journal=dest / "judge.jsonl")
    result.update(provenance)
    (dest / "diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("scope", "style", "count", "tutor", "judge_errors", "invalid")}), flush=True)


if __name__ == "__main__":
    main()
