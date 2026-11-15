"""Collect a separate KB-only tutor probe; labels are never sent to the server."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from jh_compare import ROOT, load_public, run_variant
from jh_generate_kb_facts import sections_and_hash, valid_examples
from pipelines.kimjaehak.config import VARIANTS


def build_items(source):
    meta = json.loads((source.parent / "manifest.json").read_text())
    sections, digest = sections_and_hash()
    if meta["scope"] != "provided_llm_kb_only_facts_v2" or meta["kb_sha256"] != digest:
        raise ValueError("Generation provenance differs")
    records = list(map(json.loads, source.read_text().splitlines()))
    if len(records) != len(sections) or {row["section"] for row in records} != set(sections):
        raise ValueError("Incomplete or duplicate generated sources")
    def normalized(text):
        return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())
    # Filtering happens AFTER generation, never in its API prompts.
    public_text = {normalized(m["content"]) for row in load_public() for m in row["messages"] if m["role"] == "user"}
    items, seen = [], set()
    for row in sorted(records, key=lambda r: r["section"]):
        if row["error"]:
            continue
        sid = row["section"]
        for i, example in enumerate(valid_examples(row["examples"], sid, sections[sid])):
            question = example["question"]
            key = normalized(question)
            if key in seen or key in public_text:
                continue
            seen.add(key)
            items.append({"id": f"kbprobe-{sid}-{i}", "endpoint": "tutor",
                          "messages": [{"role": "user", "content": question}],
                          "labels": {"category": "normal", "action": "answer", "evidence": [sid],
                                     "gold_facts": [example["answer_fact"]]},
                          "label_support_quote": example["quote"]})
    return items, meta


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--request-slots", type=int, default=64)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--review", type=Path, help="Completed provided-model review; required for collection")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache") or not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("Invalid source or name")
    if not 1 <= args.workers <= 64 or not 1 <= args.request_slots <= 64:
        parser.error("Invalid concurrency")
    items, meta = build_items(source)
    review_provenance = None
    if args.review:
        review_path = args.review.resolve()
        if not review_path.is_relative_to(ROOT / "experiments/.cache"):
            parser.error("Use a local provided-model review")
        review_meta = json.loads((review_path.parent / "manifest.json").read_text())
        original_dataset = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
        if review_meta["scope"] != "provided_llm_kb_fact_review" or review_meta["model"] != meta["model"] or review_meta["dataset_sha256"] != hashlib.sha256(original_dataset.encode()).hexdigest():
            parser.error("Review provenance differs")
        reviewed = list(map(json.loads, review_path.read_text().splitlines()))
        accepted = {row["id"] for row in reviewed if row["valid"] and row["error"] is None}
        if len(reviewed) != len(items) or {r["id"] for r in reviewed} != {r["id"] for r in items}:
            parser.error("Incomplete or duplicate review")
        items = [row for row in items if row["id"] in accepted]
        review_provenance = {"path": str(review_path.relative_to(ROOT)), "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(), **review_meta}
    elif not args.build_only:
        parser.error("Review the weak labels before collecting candidate responses")
    if not items:
        parser.error("No valid facts")
    root = ROOT / "experiments/.cache" / ("jh_kb_probe_" + args.name)
    root.mkdir(exist_ok=False)
    dataset = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
    (root / "derived_inputs.jsonl").write_text(dataset)
    provenance = {"scope": "provided_kb_fact_diagnostic", "count": len(items),
                  "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "dataset_sha256": hashlib.sha256(dataset.encode()).hexdigest(), "generation_model": meta["model"],
                  "kb_sha256": meta["kb_sha256"], "weak_labels": "provided_model_facts_with_exact_kb_support_quotes",
                  "score_interpretation": "Use tutor only; aggregate total includes an absent guard endpoint",
                  "review": review_provenance,
                  "public_questions_in_generation": False, "independent_source_holdout": False}
    (root / "derivation.json").write_text(json.dumps(provenance, indent=2))
    if args.build_only:
        print(json.dumps(provenance), flush=True)
        return
    args.judge_workers, args.no_judge, args.collect_only, args.endpoints = 8, False, True, ["tutor"]
    run_variant(args, args.variant, 1, items, root)
    dest = root / (args.variant + "_1")
    manifest = json.loads((dest / "manifest.json").read_text())
    manifest.update(provenance)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DERIVED_SCORING_PENDING", dest, flush=True)


if __name__ == "__main__":
    main()
