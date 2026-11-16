"""Compare KB retrieval on existing public/development data; never production."""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.kb import load_chunks
from core.llm import LLM
from core.search import Searcher, load_vectors, normalize
from pipelines.kimjaehak.round2 import SectionFusion


async def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--kb-probe", type=Path, required=True)
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        parser.error("Invalid name")
    source = args.kb_probe.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        parser.error("Use an existing local diagnostic")
    out = ROOT / "experiments/.cache" / ("jh_retrieval_" + args.name)
    out.mkdir(exist_ok=False)
    llm = LLM()
    chunks = load_chunks(ROOT / "kb")
    vectors = load_vectors(ROOT / "index/kb_embeddings.json", llm.embedding_model, chunks)
    if not llm.can_embed or vectors is None:
        raise RuntimeError("Prime the organizer KB embedding index first")
    base = Searcher(chunks, vectors)
    fused = SectionFusion(base)
    datasets = {"public_tutor": [json.loads(s) for s in (ROOT / "public/tutor.jsonl").read_text().splitlines()],
                "kb_development": [json.loads(s) for s in source.read_text().splitlines()]}
    result, saved = {}, []
    for name, all_items in datasets.items():
        items = [x for x in all_items if x["labels"].get("evidence")]
        queries = ["\n".join(m["content"] for m in row["messages"])[:6000] for row in items]
        queries_vectors = []
        for start in range(0, len(queries), 32):
            batch = await llm.embed(queries[start:start+32], timeout=60)
            if len(batch) != len(queries[start:start+32]) or any(len(v) != len(vectors[0]) or not all(math.isfinite(x) for x in v) for v in batch):
                raise ValueError("Invalid embedding response")
            queries_vectors.extend(batch)
        counts = {mode: {"top1":0,"top3":0,"errors":[]} for mode in ("bm25","dense","rrf")}
        for row, query, vector in zip(items, queries, queries_vectors):
            q = normalize(vector)
            sims = [sum(x*y for x,y in zip(q,v)) for v in vectors]
            dense = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i:-sims[i])]
            orders = {"bm25": base.search(query), "dense":dense, "rrf":fused.search(query, vector)}
            expected = set(row["labels"]["evidence"])
            for mode, found in orders.items():
                ids = list(dict.fromkeys(c.section_id for c in found))
                counts[mode]["top1"] += bool(expected.intersection(ids[:1]))
                counts[mode]["top3"] += bool(expected.intersection(ids[:3]))
                if not expected.intersection(ids[:1]):
                    counts[mode]["errors"].append({"id":row["id"],"gold":sorted(expected),"top3":ids[:3]})
            saved.append({"dataset":name,"id":row["id"],"query_sha256":hashlib.sha256(query.encode()).hexdigest(),"vector":vector})
        result[name] = {"count":len(items),"metrics":counts}
    (out / "query_vectors.jsonl").write_text("".join(json.dumps(x)+"\n" for x in saved))
    result["provenance"] = {"kb_probe_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                            "embedding_model":llm.embedding_model,"scope":"retrieval_only_development_not_holdout"}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({name:{"count":data["count"],"metrics":{m:{k:v for k,v in d.items() if k != "errors"} for m,d in data["metrics"].items()}} for name,data in result.items() if name != "provenance"}))


if __name__ == "__main__":
    asyncio.run(main())
