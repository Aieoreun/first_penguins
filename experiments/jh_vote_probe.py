"""Public-only vote diagnostics; not an HTTP/full-score/load benchmark."""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def collect(variant, workers, dest):
    from pipelines import load
    pipeline_class = load(variant)
    from core.kb import load_chunks
    from core.llm import LLM
    from core.search import Searcher
    from core.state import State
    from experiments.jh_compare import load_public
    pipeline = pipeline_class(Searcher(load_chunks(ROOT / "kb")), LLM())
    semaphore = asyncio.Semaphore(workers)
    completed = 0
    async def one(item):
        nonlocal completed
        async with semaphore:
            state = State.from_request(item)
            await pipeline.prepare(state, extra=state.candidate_answer or "")
            row = {"id": item["id"], "endpoint": item["endpoint"],
                    "labels": {key: value for key, value in item["labels"].items()
                               if key in {"category", "decision", "action", "evidence"}},
                    "label": state.label.value, "section": state.section,
                    "debug": state.debug, "errors": state.errors}
            with (dest / "votes.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 25 == 0:
                print(f"public input votes checkpointed: {completed}", flush=True)
            return row
    return await asyncio.gather(*(one(item) for item in load_public()))


def report(rows, minimum, seek_first):
    from core.schema import Label, POLICY
    from pipelines.v2.pipeline import _majority, _priority
    predictions = []
    for row in rows:
        debug = row["debug"]
        label = Label(row["label"])
        if "block_votes" in debug:
            block = _majority(debug["block_votes"]) or "none"
            required = minimum.get(block, 1) if isinstance(minimum, dict) else minimum
            if block != "none" and Counter(debug["block_votes"])[block] < required:
                block = "none"
            flags = {block: True, _majority(debug["other_votes"]) or "none": True,
                     "self_info": bool(_majority(debug["self_info_votes"])),
                     "out_of_material": debug["out_of_material"]}
            label = next((value for key, value in _priority(seek_first) if flags.get(key)), Label.NORMAL)
        predictions.append(POLICY[label])
    tp = fp = fn = cats = correct_cats = tutors = correct_actions = 0
    for row, prediction in zip(rows, predictions):
        gold = row["labels"]
        if row["endpoint"] == "tutor":
            tutors += 1
            correct_actions += prediction.action.value == gold["action"]
        else:
            blocked = prediction.decision.value == "block"
            actual = gold["decision"] == "block"
            tp += blocked and actual
            fp += blocked and not actual
            fn += not blocked and actual
            if gold["category"] != "normal":
                cats += 1
                correct_cats += prediction.category.value == gold["category"]
    return {"minimum_same_block_votes": minimum, "seek_first": seek_first,
            "guard_block_f1": 2 * tp / (2 * tp + fp + fn), "guard_false_block": fp,
            "guard_missed_block": fn, "guard_category_accuracy": correct_cats / cats,
            "tutor_action_accuracy": correct_actions / tutors}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("jh_voted", "jh_diverse"), required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name) or args.workers < 1:
        parser.error("Invalid name or workers")
    dest = ROOT / "experiments/.cache" / ("jh_votes_" + args.name)
    dest.mkdir(exist_ok=False)
    manifest = {"scope": "public_input_votes_only", "variant": args.variant,
                "workers": args.workers, "model": os.environ["OPENAI_MODEL"],
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = asyncio.run(collect(args.variant, args.workers, dest))
    (dest / "votes.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    result = {"scope": manifest["scope"], "count": len(rows),
              "stage_failure_requests": sum(bool(r["errors"]) for r in rows),
              "policies": [report(rows, minimum, seek) for minimum in (1, 2, 3) for seek in (True, False)]}
    (dest / "diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
