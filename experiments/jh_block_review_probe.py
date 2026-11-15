"""Public-only review of recorded block decisions; never a full guard score.

Gold labels are used only in final metrics. Category after release is a
placeholder because this diagnostic does not rerun the other input checks.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def run(source, dest, workers):
    from pipelines import load
    pipeline_class = load("jh_reconsider")
    from core.kb import load_chunks
    from core.llm import LLM
    from core.schema import parse_label
    from core.search import Searcher
    from core.state import State
    from experiments.jh_compare import load_public
    items = load_public(endpoints=("guard",))
    prior_rows = list(map(json.loads, (source / "responses.jsonl").read_text().splitlines()))
    prior = {row["id"]: row["response"] for row in prior_rows}
    if len(prior_rows) != len(prior):
        raise ValueError("Duplicate source responses")
    for item in items:
        prediction = prior.get(item["id"], {}) or {}
        inp = prediction.get("input", {})
        if parse_label(inp.get("category", "")) is None or inp.get("decision") not in {"allow", "block"}:
            raise ValueError("Complete public input predictions are required")
    pipe = pipeline_class(Searcher(load_chunks(ROOT / "kb")), LLM())
    semaphore = asyncio.Semaphore(workers)
    async def one(item):
        inp = prior[item["id"]]["input"]
        before = inp["decision"]
        state = State(messages=item["messages"], label=parse_label(inp["category"]))
        started = time.monotonic()
        if before == "block":
            async with semaphore:
                started = time.monotonic()
                await pipe._review_block(state)
        after = state.decision.value if before == "block" else before
        row = {"id": item["id"], "before": before, "after": after,
               "gold": item["labels"]["decision"], "errors": state.errors,
               "seconds": round(time.monotonic() - started, 3)}
        with (dest / "reviews.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return row
    rows = await asyncio.gather(*(one(item) for item in items))
    def metrics(field):
        tp = sum(row[field] == row["gold"] == "block" for row in rows)
        fp = sum(row[field] == "block" and row["gold"] == "allow" for row in rows)
        fn = sum(row[field] == "allow" and row["gold"] == "block" for row in rows)
        return {"block_f1": 2 * tp / (2 * tp + fp + fn), "false_block": fp, "missed_block": fn}
    return {"scope":"public_block_review_only", "count":len(rows),
            "reviewed":sum(row["before"] == "block" for row in rows),
            "stage_failure_requests":sum(bool(row["errors"]) for row in rows),
            "before":metrics("before"), "after":metrics("after")}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache") or not re.fullmatch(r"[A-Za-z0-9_-]+", args.name) or args.workers < 1:
        parser.error("Only local public recordings and a valid name/worker count are supported")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["model"] != os.environ["OPENAI_MODEL"]:
        parser.error("The model must match the source run")
    dest = ROOT / "experiments/.cache" / ("jh_blockreview_" + args.name)
    dest.mkdir(exist_ok=False)
    metadata = {"scope":"public_block_review_only", "source":str(source.relative_to(ROOT)),
                "source_response_sha256":hashlib.sha256((source / "responses.jsonl").read_bytes()).hexdigest(),
                "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "workers":args.workers,"model":os.environ["OPENAI_MODEL"]}
    (dest / "manifest.json").write_text(json.dumps(metadata, indent=2))
    result = asyncio.run(run(source, dest, args.workers))
    (dest / "diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
