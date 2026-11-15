"""PUBLIC-only component diagnostic: fix our own input predictions, vary output.

This is not an end-to-end HTTP or hidden evaluation. Gold labels never populate
pipeline state; only previously generated public input predictions are replayed.
One fresh process per variant; BM25-only sources with top1 section are supported.
"""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.jh_compare import evaluator_hashes, load_public, score_rows
from core.kb import load_chunks
from core.llm import LLM
from core.schema import Decision, parse_label
from core.search import Searcher
from pipelines import load
from pipelines.kimjaehak.config import VARIANTS


def predicted_state(item, response):
    from pipelines.kimjaehak.pipeline import PeaceState
    label = parse_label(response["input"]["category"])
    if label is None:
        raise ValueError("Source has an invalid input category")
    return PeaceState(messages=item["messages"], candidate_answer=item.get("candidate_answer"),
                      label=label, peace_decision=Decision(response["input"]["decision"]))


async def collect(pipeline, items, prior, workers, dest):
    sem = asyncio.Semaphore(workers)
    with (dest / "responses.jsonl").open("w") as stream:
        async def one(item):
            old = prior[item["id"]]["response"]
            state = predicted_state(item, old)
            start = time.monotonic()
            if state.has_candidate:
                async with sem:
                    start = time.monotonic()
                    await pipeline._search(state, extra=state.candidate_answer or "")
                    state.section = state.chunks[0].section_id if state.chunks else None
                    await pipeline._check_output(state)
            response = state.to_guard_response()
            if response["input"] != old["input"]:
                raise ValueError("Input predictions changed during output-only replay")
            row = {"id": item["id"], "endpoint": "guard", "response": response,
                   "seconds": round(time.monotonic() - start, 3), "error": None,
                   "stage_errors": state.errors}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            return row
        return await asyncio.gather(*(one(item) for item in items))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--variant", choices=(*VARIANTS, "v4"), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name) or args.workers < 1:
        ap.error("Invalid name or workers")
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        ap.error("Only local public recordings are supported")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("embedding_configured"):
        ap.error("Only BM25 source runs can be faithfully replayed")
    if manifest["variant"] not in ("jh_baseline", "v4"):
        ap.error("Use a baseline/v4 source with unchanged top1 search")
    if manifest["evaluator_hashes"] != evaluator_hashes():
        ap.error("Evaluator hashes changed")
    # Configure the variant before predicted_state imports shared pipeline modules.
    pipeline_class = load(args.variant)
    items = load_public(endpoints=("guard",))
    prior = {r["id"]: r for r in map(json.loads, (source / "responses.jsonl").read_text().splitlines())}
    for item in items:
        if not prior.get(item["id"], {}).get("response"):
            ap.error("Source lacks complete public guard predictions")
        predicted_state(item, prior[item["id"]]["response"])
    dest = ROOT / "experiments/.cache" / ("jh_output_" + args.name)
    dest.mkdir(exist_ok=False)
    metadata = {"scope": "public_output_component_only", "variant": args.variant,
        "source": str(source.relative_to(ROOT)), "source_manifest": manifest,
        "source_response_sha256": hashlib.sha256((source / "responses.jsonl").read_bytes()).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "workers": args.workers, "model": os.environ["OPENAI_MODEL"], "evaluator_hashes": evaluator_hashes()}
    (dest / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    async def run():
        pipeline = pipeline_class(Searcher(load_chunks(ROOT / "kb")), LLM())
        return await collect(pipeline, items, prior, args.workers, dest)
    rows = asyncio.run(run())
    result = score_rows(items, rows, args.workers, no_judge=True)
    result["score_scope"] = "public_output_component_only"
    result["stage_failure_requests"] = sum(bool(r["stage_errors"]) for r in rows)
    if metadata["evaluator_hashes"] != evaluator_hashes():
        raise RuntimeError("Evaluator changed")
    (dest / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("guard", "invalid", "guard_details", "stage_failure_requests")}), flush=True)
    print("RESULT_DIR", dest, flush=True)


if __name__ == "__main__":
    main()
