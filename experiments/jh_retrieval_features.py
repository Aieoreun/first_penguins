"""Retrieval-only development screen using KB and already embedded public queries."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.kb import load_chunks
from core.search import BM25, Searcher, load_vectors


def main():
    root = ROOT / "experiments/.cache/jh_retrieval_r2_initial"
    target = root / "feature_screen.json"
    if target.exists():
        raise ValueError("Refusing to overwrite diagnostic")
    chunks = load_chunks(ROOT / "kb")
    vectors = np.asarray(load_vectors(ROOT / "index/kb_embeddings.json", "text-embedding-3-small", chunks))
    base = Searcher(chunks)
    ids = base.section_ids()
    sections = BM25([base.section_text(sid) for sid in ids])
    cached = {(r["dataset"], r["id"]): np.asarray(r["vector"]) for r in
              map(json.loads, (root / "query_vectors.jsonl").read_text().splitlines())}
    sources = {"public_tutor": ROOT / "public/tutor.jsonl",
               "kb_development": ROOT / "experiments/.cache/jh_kb_probe_r2_base_off/derived_inputs.jsonl"}
    results = {}
    for dataset, source in sources.items():
        rows = [r for r in map(json.loads, source.read_text().splitlines()) if r["labels"].get("evidence")]
        counts = {}
        for row in rows:
            query = "\n".join(m["content"] for m in row["messages"])
            last = next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user")
            lexical = np.asarray(base.bm25.scores(query))
            latest = np.asarray(base.bm25.scores(last))
            vector = cached[dataset, row["id"]]
            dense = vectors @ (vector / max(np.linalg.norm(vector), 1e-12))
            def norm(x):
                return x / max(x.max(), 1e-12)
            modes = {"chunk_all": lexical, "chunk_last": latest,
                     "chunk_last_history": .75 * norm(latest) + .25 * norm(lexical),
                     "chunk_dense02": .8 * norm(lexical) + .2 * norm(dense),
                     "chunk_dense05": .5 * norm(lexical) + .5 * norm(dense)}
            orders = {name: list(dict.fromkeys(chunks[i].section_id for i in
                      sorted(range(len(chunks)), key=lambda i: -scores[i]))) for name, scores in modes.items()}
            section_scores = sections.scores(query)
            orders["section_all"] = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -section_scores[i])]
            for mode, order in orders.items():
                hit = set(row["labels"]["evidence"])
                entry = counts.setdefault(mode, {"top1": 0, "top3": 0, "errors": []})
                entry["top1"] += bool(hit.intersection(order[:1]))
                entry["top3"] += bool(hit.intersection(order[:3]))
                if not hit.intersection(order[:1]):
                    entry["errors"].append({"id": row["id"], "gold": sorted(hit), "top3": order[:3]})
        results[dataset] = {"count": len(rows), "metrics": counts}
    results["scope"] = "retrieval_only_development_screen_not_holdout"
    target.write_text(json.dumps(results, indent=2))
    for name, result in results.items():
        if isinstance(result, dict):
            print(name, {k: {m: v for m, v in x.items() if m != "errors"} for k, x in result["metrics"].items()})


if __name__ == "__main__":
    main()
