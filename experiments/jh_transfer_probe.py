"""Input-only transfer diagnostic on untrained, public-derived paraphrases.

The labels are inherited from public sources and may contain generation noise.
This is neither the official hidden evaluation nor independent new source data.
Runtime requests contain only messages and random session IDs, never labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

from jh_compare import ROOT, load_public, run_variant, score_rows
from pipelines.kimjaehak.config import VARIANTS

BLOCKED = {"academic_dishonesty","off_topic","harmful_content","jailbreak_injection"}


def normalized(text):
    return re.sub(r"[\W_]+","",unicodedata.normalize("NFKC",text).casefold())


def build_items(source, alternative=False):
    used = set()
    if alternative:
        primary,_ = build_items(source)
        used = {json.dumps(row["messages"],ensure_ascii=False,sort_keys=True) for row in primary}
    metadata = json.loads((source.parent / "manifest.json").read_text())
    expected = {name:hashlib.sha256((ROOT / "public" / name).read_bytes()).hexdigest()
                for name in ("tutor.jsonl","guard.jsonl")}
    if metadata.get("scope") != "provided_llm_public_paraphrases" or metadata["public_hashes"] != expected:
        raise ValueError("Public derivation provenance mismatch")
    records = list(map(json.loads,source.read_text().splitlines()))
    generated = {row["source_id"]:row for row in records}
    if len(generated) != len(records) or set(generated) != set(metadata["source_ids"]):
        raise ValueError("Incomplete or duplicate generated sources")
    public = load_public()
    originals = {normalized(item["messages"][-1]["content"]) for item in public}
    items, seen = [], set()
    for item in public:
        row = generated.get(item["id"])
        category = item["labels"]["category"]
        if not row or row["error"] or category == "low_signal_input":
            continue
        if row["category"] != category:
            raise ValueError("Generated category changed")
        for index,text in enumerate(row["paraphrases"]):
            if normalized(text) in originals:
                continue
            messages = [dict(m) for m in item["messages"]]
            if messages[-1]["role"] != "user":
                raise ValueError("Unexpected source conversation")
            messages[-1]["content"] = text
            key = json.dumps(messages,ensure_ascii=False,sort_keys=True)
            if key in seen or key in used:
                continue
            seen.add(key)
            items.append({"id":item["id"]+f"-p{index}","source_id":item["id"],
                "endpoint":"guard","messages":messages,
                "labels":{"category":category,"decision":"block" if category in BLOCKED else "allow"}})
            break
    return items, metadata


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env",override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--variant",choices=VARIANTS,required=True)
    parser.add_argument("--name",required=True)
    parser.add_argument("--workers",type=int,default=64)
    parser.add_argument("--judge-workers",type=int,default=8)
    parser.add_argument("--request-slots",type=int,default=64)
    parser.add_argument("--embedding-mode", choices=("env", "off"), default="env")
    parser.add_argument("--build-only",action="store_true")
    parser.add_argument("--alternative",action="store_true",help="Use expressions excluded from the primary transfer set")
    parser.add_argument("--full",action="store_true",help="Retain source endpoints and inherited answer/evidence labels")
    parser.add_argument("--endpoints", nargs="+", choices=("tutor", "guard"),
                        help="With --full, select endpoint diagnostics; total is not a full-set score")
    parser.add_argument("--collect-only",action="store_true",help="Defer full derived scoring until collection comparisons finish")
    args = parser.parse_args()
    if not 1 <= args.judge_workers <= 64:
        parser.error("Invalid judge concurrency")
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache") or not re.fullmatch(r"[A-Za-z0-9_-]+",args.name) or not 1 <= args.workers <= 64 or not 1 <= args.request_slots <= 64:
        parser.error("Invalid source, name or concurrency")
    items, generation = build_items(source,args.alternative)
    if args.collect_only and not args.full:
        parser.error("Input-only diagnostics need no deferred judge")
    if args.endpoints and (not args.full or len(set(args.endpoints)) != len(args.endpoints)):
        parser.error("Endpoint selection needs --full and distinct endpoints")
    args.endpoints = args.endpoints or (["tutor", "guard"] if args.full else ["guard"])
    if args.full:
        originals = {row["id"]:row for row in load_public()}
        items = [{**originals[row["source_id"]],"id":row["id"],"source_id":row["source_id"],
                  "messages":row["messages"]} for row in items]
        items = [item for item in items if item["endpoint"] in args.endpoints]
    if not items:
        parser.error("No valid derived inputs")
    root = ROOT / "experiments/.cache" / ("jh_transfer_" + args.name)
    root.mkdir(exist_ok=False)
    dataset = "".join(json.dumps(item,ensure_ascii=False)+"\n" for item in items)
    (root / "derived_inputs.jsonl").write_text(dataset)
    provenance = {"scope":"public_paraphrase_full_diagnostic" if args.full else "public_paraphrase_input_diagnostic","count":len(items),
        "source":str(source.relative_to(ROOT)),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_sha256":hashlib.sha256(dataset.encode()).hexdigest(),"generation_model":generation["model"],
        "inherited_labels":True,"independent_source_holdout":False,
        "paraphrase_partition":"alternative" if args.alternative else "primary",
        "score_scope":"+".join(args.endpoints)}
    (root / "derivation.json").write_text(json.dumps(provenance,indent=2))
    if args.build_only:
        print(json.dumps(provenance),flush=True)
        return
    # Collection uses the same HTTP transport and server as the full comparison.
    # Only full diagnostics need the original answer judge.
    deferred = args.collect_only
    args.judge_workers, args.no_judge, args.collect_only = (args.judge_workers if args.full else 1), not args.full, True
    run_variant(args,args.variant,1,items,root)
    dest = root / (args.variant + "_1")
    manifest = json.loads((dest / "manifest.json").read_text())
    manifest.update(provenance)
    (dest / "manifest.json").write_text(json.dumps(manifest,indent=2))
    if deferred:
        print("DERIVED_SCORING_PENDING",dest,flush=True)
        return
    rows = list(map(json.loads,(dest / "responses.jsonl").read_text().splitlines()))
    result = score_rows(items,rows,args.judge_workers,no_judge=not args.full,journal=dest / "judge.jsonl")
    if args.full:
        result.update(provenance)
        result["runtime"] = json.loads((dest / "runtime.json").read_text())
        (dest / "derived_result.json").write_text(json.dumps(result,indent=2))
        print(json.dumps({key:result[key] for key in ("scope","count","total","tutor","guard","judge_errors","invalid")}),flush=True)
        return
    diagnostic = {**provenance,"variant":args.variant,"invalid":result["invalid"],
                  "input_metrics":result["guard_details"],"runtime":json.loads((dest / "runtime.json").read_text())}
    (dest / "diagnostic.json").write_text(json.dumps(diagnostic,indent=2))
    print(json.dumps({"count":len(items),"variant":args.variant,"invalid":result["invalid"],
        **{key:result["guard_details"][key] for key in ("block_f1","category_accuracy","false_block","missed_block")}}),flush=True)


if __name__ == "__main__":
    main()
