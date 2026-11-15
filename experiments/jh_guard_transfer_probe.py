"""Replay own input predictions on novel, provided-model PUBLIC guard answers."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.jh_compare import evaluator_hashes,score_rows
from experiments.jh_output_replay import collect
from experiments.jh_guard_paraphrases import normalized
from pipelines.kimjaehak.config import VARIANTS


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/".env",override=True)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--variant",choices=VARIANTS,required=True)
    ap.add_argument("--name",required=True)
    ap.add_argument("--workers",type=int,default=16)
    args=ap.parse_args()
    source,predictions=args.source.resolve(),args.predictions.resolve()
    if (not re.fullmatch(r"[A-Za-z0-9_-]+",args.name) or not 1<=args.workers<=64
            or not all(p.is_relative_to(ROOT/"experiments/.cache") for p in (source,predictions))):
        ap.error("Invalid paths/name/workers")
    raw=ROOT/"public/guard.jsonl";originals={r["id"]:r for r in map(json.loads,raw.read_text().splitlines())}
    metadata=json.loads((source.parent/"manifest.json").read_text())
    if (metadata.get("scope")!="provided_llm_public_guard_answer_paraphrases"
            or metadata["source_sha256"]!=hashlib.sha256(raw.read_bytes()).hexdigest()
            or metadata["model"]!=os.environ["OPENAI_MODEL"]):
        ap.error("Generated data provenance differs")
    generated=list(map(json.loads,source.read_text().splitlines()))
    ids={r["source_id"] for r in generated}
    if ids!=set(metadata["source_ids"]) or len(ids)!=len(generated):ap.error("Incomplete or duplicate generation")
    prior_manifest=json.loads((predictions.parent/"manifest.json").read_text())
    if prior_manifest["evaluator_hashes"]!=evaluator_hashes():ap.error("Evaluator changed")
    if prior_manifest.get("scope") or prior_manifest.get("count")!=400:
        ap.error("Use complete original public predictions")
    prior={r["id"]:r for r in map(json.loads,predictions.read_text().splitlines())}
    items=[]
    original_texts={normalized(r["candidate_answer"]) for r in originals.values() if r.get("candidate_answer")}
    for row in sorted(generated,key=lambda r:r["source_id"]):
        if row["error"] is not None:continue
        original=originals[row["source_id"]]
        if (normalized(row["candidate_answer"]) in original_texts
                or any(s not in row["candidate_answer"] for s in row["unsupported"])
                or bool(row["unsupported"])!=bool(original["labels"]["unsupported"])):
            ap.error("Generated annotations are inconsistent")
        items.append({**original,"candidate_answer":row["candidate_answer"],
                      "labels":{**original["labels"],"unsupported":row["unsupported"]}})
    if not items:ap.error("No valid novel answers")
    from pipelines import load
    pipeline_class=load(args.variant)
    dest=ROOT/"experiments/.cache"/f"jh_guard_transfer_{args.name}";dest.mkdir(exist_ok=False)
    data="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in items)
    (dest/"inputs.jsonl").write_text(data)
    manifest={"scope":"public_guard_answer_paraphrase_component","count":len(items),"variant":args.variant,
              "source":str(source.relative_to(ROOT)),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "dataset_sha256":hashlib.sha256(data.encode()).hexdigest(),"generation":metadata,
              "predictions":str(predictions.relative_to(ROOT)),"predictions_sha256":hashlib.sha256(predictions.read_bytes()).hexdigest(),
              "model":os.environ["OPENAI_MODEL"],"workers":args.workers,"evaluator_hashes":evaluator_hashes(),
              "commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "used_in_training":False,"independent_source_holdout":False,"input_predictions_fixed":True}
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    async def run():
        from core.kb import load_chunks
        from core.llm import LLM
        from core.search import Searcher
        llm=LLM()
        try:return await collect(pipeline_class(Searcher(load_chunks(ROOT/"kb")),llm),items,prior,args.workers,dest)
        finally:await llm._chat.close()
    rows=asyncio.run(run());result=score_rows(items,rows,8,no_judge=True)
    result.update(scope=manifest["scope"],count=len(items),stage_failure_requests=sum(bool(r["stage_errors"]) for r in rows))
    if manifest["evaluator_hashes"]!=evaluator_hashes():raise RuntimeError("Evaluator changed")
    (dest/"diagnostic.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("guard","guard_details","invalid","stage_failure_requests")}),flush=True)


if __name__=="__main__":main()
