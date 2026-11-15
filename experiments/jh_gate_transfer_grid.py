"""Compare category regularization on provided public paraphrases, not a holdout."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from pipelines.kimjaehak.learned_gate import train_public_classifier

BLOCKED = {"harmful_content", "academic_dishonesty", "jailbreak_injection", "off_topic"}


def read_derived(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT / "experiments/.cache"):
        raise ValueError("Use recorded public-derived data")
    meta = json.loads((path.parent / "derivation.json").read_text())
    if meta["scope"] != "public_paraphrase_input_diagnostic" or hashlib.sha256(path.read_bytes()).hexdigest() != meta["dataset_sha256"]:
        raise ValueError("Source scope or hash mismatch")
    return list(map(json.loads, path.read_text().splitlines())), meta


def last(row):
    return next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user")


def run(dataset, source, alternative):
    rows, meta = read_derived(dataset)
    alternate, alt_meta = read_derived(alternative)
    source = source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        raise ValueError("Use a recorded public diagnostic")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("scope") != meta["scope"] or manifest.get("dataset_sha256") != meta["dataset_sha256"]:
        raise ValueError("Recorded predictions do not match the derived dataset")
    prior = {r["id"]: r["response"]["input"]["category"] for r in map(json.loads, (source / "responses.jsonl").read_text().splitlines())}
    y = np.array([r["labels"]["category"] for r in rows])
    alt_y = np.array([r["labels"]["category"] for r in alternate])
    base = np.array([prior[r["id"]] for r in rows], dtype=object)
    report = {"scope": "public_derived_category_grid", "independent_source_holdout": False,
              "primary": meta, "alternative": alt_meta, "baseline_source": str(source.relative_to(ROOT)),
              "public_hashes": {f"{name}.jsonl": hashlib.sha256((ROOT / "public" / f"{name}.jsonl").read_bytes()).hexdigest()
                                for name in ("tutor", "guard")},
              "baseline_sha256": hashlib.sha256((source / "responses.jsonl").read_bytes()).hexdigest(), "grid": []}
    for c in (100, 300, 1000, 3000, 10000):
        vectorizer, model = train_public_classifier(c)
        p = model.predict_proba(vectorizer.transform([last(r) for r in rows]))
        pred, confidence = model.classes_[p.argmax(axis=1)], p.max(axis=1)
        ap = model.predict_proba(vectorizer.transform([last(r) for r in alternate]))
        alt_pred, alt_confidence = model.classes_[ap.argmax(axis=1)], ap.max(axis=1)
        for threshold in (.95, .98, .995):
            mask = (confidence >= threshold) & (pred != "harmful_content")
            alt_mask = (alt_confidence >= threshold) & (alt_pred != "harmful_content")
            combined = np.where(mask, pred, base)
            truth_blocked = np.array([v in BLOCKED for v in y])
            pred_blocked = np.array([v in BLOCKED for v in combined])
            report["grid"].append({"C": c, "threshold": threshold, "coverage": int(mask.sum()),
                "hits": int(sum(mask & (pred == y))), "alternative_coverage": int(alt_mask.sum()),
                "alternative_hits": int(sum(alt_mask & (alt_pred == alt_y))),
                "combined_block_f1": float(f1_score(truth_blocked, pred_blocked)),
                "combined_non_normal_accuracy": float(np.mean(combined[y != "normal"] == y[y != "normal"])),
                "false_block": int(sum(~truth_blocked & pred_blocked)), "missed_block": int(sum(truth_blocked & ~pred_blocked))})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    with threadpool_limits(limits=1):
        report = run(args.dataset, args.source, args.alternative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["grid"]:
        print(json.dumps(row), flush=True)
