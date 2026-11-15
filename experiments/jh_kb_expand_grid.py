"""Compare KB-only generated questions as an auxiliary lexical index.

Public evidence labels measure retrieval only. All public data has already been
inspected; deterministic halves are diagnostics, not pristine held-out tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.kb import load_chunks
from core.search import BM25, Searcher


def run(source):
    searcher = Searcher(load_chunks(ROOT / "kb"))
    sections = {sid:searcher.section_text(sid) for sid in searcher.section_ids()}
    metadata = json.loads((source.parent / "manifest.json").read_text())
    digest = hashlib.sha256(json.dumps(sections,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if metadata.get("scope") != "provided_llm_kb_only_questions" or metadata["source_sha256"] != digest:
        raise ValueError("KB expansion provenance mismatch")
    data = list(map(json.loads,source.read_text().splitlines()))
    if len(data) != len(sections) or {row["section"] for row in data} != set(sections):
        raise ValueError("Incomplete or duplicate expansion sections")
    queries = [(row["section"],text) for row in data if row["error"] is None for text in row["questions"]]
    by_section = {sid:[] for sid in sections}
    for sid,text in queries:
        by_section[sid].append(text)
    order = list(sections)
    index = {"max":BM25([text for _,text in queries]),
             "joined":BM25(["\n".join(by_section[sid]) for sid in order])}
    question_sections = [sid for sid,_ in queries]
    public = ROOT / "public/tutor.jsonl"
    rows = [row for row in map(json.loads,public.read_text().splitlines()) if len(row["labels"].get("evidence") or []) == 1]
    prepared = []
    for row in rows:
        all_text = "\n".join(m["content"] for m in row["messages"])
        last = next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user")
        raw = searcher.bm25.scores(all_text)
        base = {sid:0. for sid in order}
        for chunk,score in zip(searcher.chunks,raw):
            base[chunk.section_id] = max(base[chunk.section_id],score)
        scale = max(base.values()) or 1
        base = {sid:value/scale for sid,value in base.items()}
        generated = {}
        for mode,question_text in (("all",all_text),("last",last)):
            for aggregation,model in index.items():
                scores = {sid:0. for sid in order}
                ids = question_sections if aggregation == "max" else order
                for sid,score in zip(ids,model.scores(question_text)):
                    scores[sid] = max(scores[sid],score)
                scale = max(scores.values()) or 1
                generated[(mode,aggregation)] = {sid:value/scale for sid,value in scores.items()}
        prepared.append((row,base,generated))
    reports = []
    for mode in ("all","last"):
        for aggregation in ("max","joined"):
            for weight in (0,.1,.2,.3,.5,.75,1,1.5,2,3):
                hits, halves, sizes = [], [0,0], [0,0]
                for row,base,generated in prepared:
                    scores = {sid:base[sid]+weight*generated[(mode,aggregation)][sid] for sid in order}
                    pred = max(order,key=lambda sid:scores[sid]) if max(scores.values()) else ""
                    hit = pred == row["labels"]["evidence"][0]
                    part = int(hashlib.sha256(row["id"].encode()).hexdigest()[-1],16)%2
                    halves[part] += hit; sizes[part] += 1; hits.append(hit)
                reports.append({"query":mode,"aggregation":aggregation,"weight":weight,
                    "hits":sum(hits),"count":len(hits),"half_hits":halves,"half_sizes":sizes})
    reports.sort(key=lambda r:(-r["hits"],r["weight"]))
    return {"scope":"public_kb_expansion_retrieval","source":str(source.relative_to(ROOT)),
            "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"model":metadata["model"],
            "public_sha256":hashlib.sha256(public.read_bytes()).hexdigest(),"reports":reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        parser.error("Use a recorded KB-only expansion")
    result = run(source)
    (source.parent / "retrieval_grid.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"top":result["reports"][:5],"baseline":next(row for row in result["reports"] if row["weight"]==0)}))
