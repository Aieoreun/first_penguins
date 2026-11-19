"""Review frozen public tutor drafts; never substitute gold input/evidence labels.

Component diagnostic only. The production candidate still has to pass a fresh
HTTP comparison with its original request deadline and concurrent workload.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jh_compare import evaluator_hashes, load_public, score_rows
from core.kb import load_chunks
from core.llm import LLM
from core.search import Searcher
from pipelines.kimjaehak.answer_review import REVIEW_SYSTEM, reviewed_answer


async def collect(items, prior, workers, dest):
    llm, searcher = LLM(), Searcher(load_chunks(ROOT / "kb"))
    allowed = set(searcher.section_ids())
    slots = asyncio.Semaphore(workers)
    with (dest / "responses.jsonl").open("w") as stream:
        async def one(item):
            old = prior[item["id"]]
            row = copy.deepcopy(old)
            response = row.get("response") or {}
            citations = response.get("citations")
            row.update(source_seconds=old["seconds"], seconds=0, review_attempted=False,
                       review_changed=False, review_error=None)
            if (not old["error"] and response.get("action") == "answer"
                    and isinstance(citations,list) and len(citations)==1 and citations[0] in allowed
                    and isinstance(response.get("answer"),str) and response["answer"].strip()):
                async with slots:
                    start = time.monotonic()
                    row["review_attempted"] = True
                    try:
                        answer = await reviewed_answer(llm, searcher.section_text(citations[0]),
                                                       item["messages"], response["answer"], 7)
                        if answer is not None:
                            response["answer"] = f"{answer} ({citations[0]})"
                            row["review_changed"] = True
                    except Exception as exc:
                        row["review_error"] = type(exc).__name__
                    row["seconds"] = round(time.monotonic()-start, 3)
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            stream.flush()
            return row
        return await asyncio.gather(*(one(item) for item in items))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/".env", override=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+",args.name) or args.workers < 1:
        ap.error("Invalid name or workers")
    source = args.source.resolve()
    if not source.is_relative_to(ROOT/"experiments/.cache"):
        ap.error("Only public local recordings are supported")
    manifest = json.loads((source/"manifest.json").read_text())
    result = json.loads((source/"result.json").read_text())
    if manifest.get("scope") or manifest["count"]!=400 or manifest["evaluator_hashes"]!=evaluator_hashes():
        ap.error("Use an unchanged full bundled-public source")
    if manifest["model"] != os.environ.get("OPENAI_MODEL"):
        ap.error("Model differs from the recorded source")
    records = list(map(json.loads,(source/"responses.jsonl").read_text().splitlines()))
    prior = {row["id"]:row for row in records}
    public = load_public()
    if len(records)!=400 or set(prior)!={item["id"] for item in public}:
        ap.error("Incomplete public responses")
    items = [item for item in public if item["endpoint"]=="tutor"]
    dest = ROOT/"experiments/.cache"/("jh_review_"+args.name)
    dest.mkdir(exist_ok=False)
    metadata = {"scope":"public_frozen_draft_review_component_not_http", "source":str(source.relative_to(ROOT)),
        "source_manifest":manifest,"source_response_sha256":hashlib.sha256((source/"responses.jsonl").read_bytes()).hexdigest(),
        "source_tutor_score":result["tutor"],"workers":args.workers,"count":len(items),
        "prompt_sha256":hashlib.sha256(REVIEW_SYSTEM.encode()).hexdigest(),"evaluator_hashes":evaluator_hashes(),
        "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "input_and_citations":"frozen_own_predictions_never_gold",
        "runtime_limit":"revision_only_7s_not_end_to_end", "review_categories":"own_answer_action_with_one_valid_citation"}
    (dest/"manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    rows = asyncio.run(collect(items,prior,args.workers,dest))
    # Reuse only identical public judging calls within this recorded diagnostic.
    # No production request is recorded or cached by this tool.
    shutil.copyfile(source/"judge.jsonl",dest/"judge.jsonl")
    baseline = score_rows(items,[prior[item["id"]] for item in items],args.workers,journal=dest/"judge.jsonl")
    revised = score_rows(items,rows,args.workers,journal=dest/"judge.jsonl")
    report = {**metadata,"baseline_tutor":baseline["tutor"],"revised_tutor":revised["tutor"],
              "changed":sum(row["review_changed"] for row in rows),
              "attempted":sum(row["review_attempted"] for row in rows),
              "review_errors":sum(row["review_error"] is not None for row in rows),
              "baseline":baseline,"revised":revised}
    if evaluator_hashes()!=metadata["evaluator_hashes"]:
        raise ValueError("Evaluator changed")
    (dest/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ("scope","baseline_tutor","revised_tutor","attempted","changed","review_errors")}),flush=True)


if __name__=="__main__":
    main()
