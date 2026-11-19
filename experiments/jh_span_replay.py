"""Isolate deterministic span refinement on one recorded novel-answer probe."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.jh_compare import evaluator_hashes,score_rows
from pipelines.kimjaehak.span_tagger import SpanTagger,public_rows,refine_spans


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    args=ap.parse_args();source=args.source.resolve()
    if not source.is_relative_to(ROOT/"experiments/.cache"):ap.error("Use a local public diagnostic")
    out=source/"span_effect.json"
    if out.exists():ap.error("Effect already recorded")
    manifest=json.loads((source/"manifest.json").read_text())
    if manifest.get("scope")!="public_guard_answer_paraphrase_component":ap.error("Unexpected source scope")
    if manifest["evaluator_hashes"]!=evaluator_hashes():ap.error("Evaluator changed")
    if manifest["dataset_sha256"]!=hashlib.sha256((source/"inputs.jsonl").read_bytes()).hexdigest():
        ap.error("Dataset changed")
    items=list(map(json.loads,(source/"inputs.jsonl").read_text().splitlines()))
    old=list(map(json.loads,(source/"responses.jsonl").read_text().splitlines()))
    rows=deepcopy(old);by_id={r["id"]:r for r in items}
    if {r["id"] for r in rows}!=set(by_id) or len(rows)!=len(items):ap.error("Incomplete/duplicate responses")
    model=SpanTagger(public_rows(),100);effects=[]
    for row in rows:
        candidate=by_id[row["id"]]["candidate_answer"]
        original=row["response"]["output"]["unsupported"]
        predicted,confidence=model.predict(candidate)
        spans,changed=refine_spans(candidate,original,predicted,confidence,.95)
        row["response"]["output"]["unsupported"]=spans
        effects.append({"id":row["id"],"confidence":confidence,"applied":changed,
                        "changed":spans!=original,"grounded_preserved":True})
    def metrics(responses):
        result=score_rows(items,responses,8,no_judge=True)
        return {"guard":result["guard"],"invalid":result["invalid"],"guard_details":result["guard_details"]}
    result={"scope":"novel_guard_answer_fixed_output_span_effect","source":str(source.relative_to(ROOT)),
            "dataset_sha256":manifest["dataset_sha256"],
            "responses_sha256":hashlib.sha256((source/"responses.jsonl").read_bytes()).hexdigest(),
            "count":len(items),"no_new_llm_calls":True,"C":100,"threshold":.95,
            "baseline":metrics(old),"refined":metrics(rows),"applications":sum(e["applied"] for e in effects),
            "changed":sum(e["changed"] for e in effects),"items":effects}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="items"}),flush=True)


if __name__=="__main__":main()
