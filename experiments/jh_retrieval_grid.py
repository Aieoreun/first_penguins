"""Offline public retrieval sensitivity; no LLM calls, no pretrained weights.

All public data has already been inspected. The deterministic partitions below
are diagnostics, not an untouched holdout or a claim about private evaluation.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from core.kb import load_chunks
from core.search import BM25, tokens
from experiments.split import which


def run():
    chunks = load_chunks(ROOT / "kb")
    items = [p for p in map(json.loads, (ROOT / "public/tutor.jsonl").read_text().splitlines())
             if p["labels"].get("evidence")]
    base = BM25([c.text for c in chunks])
    title = BM25([c.title + " " + c.heading for c in chunks])
    vocab = {word: i for i, word in enumerate(base.idf)}
    q = np.zeros((len(items), len(vocab)))
    last = np.zeros_like(q)
    for i, item in enumerate(items):
        for word in tokens("\n".join(m["content"] for m in item["messages"])):
            if word in vocab:
                q[i, vocab[word]] = 1
        for word in tokens(next(m["content"] for m in reversed(item["messages"]) if m["role"] == "user")):
            if word in vocab:
                last[i, vocab[word]] = 1
    title_scores = np.array([title.scores("\n".join(m["content"] for m in p["messages"])) for p in items])
    gold = np.array([[c.section_id in p["labels"]["evidence"] for c in chunks] for p in items])
    dev = np.array([which(p["id"]) == "dev" for p in items])
    reports = []
    for k1 in (0.3, 0.8, 1.5, 3):
        for b in (0, 0.25, 0.5, 0.75, 1):
            doc = np.zeros((len(chunks), len(vocab)))
            for i, (tf, length) in enumerate(zip(base.tf, base.length)):
                for word, f in tf.items():
                    doc[i, vocab[word]] = base.idf[word] * f * (k1 + 1) / (f + k1 * (1 - b + b * length / base.avg))
            scores, latest = q @ doc.T, last @ doc.T
            for last_weight in (0, 0.25, 0.5, 1, 2):
                for title_weight in (0, 0.5, 1):
                    ranking = np.argsort(-(scores + last_weight * latest + title_weight * title_scores), axis=1, kind="stable")
                    hits = gold[np.arange(len(items)), ranking[:, 0]]
                    reports.append({"k1": k1, "b": b, "last_weight": last_weight,
                                    "title_weight": title_weight, "hits": int(hits.sum()),
                                    "dev_hits": int(hits[dev].sum()), "other_hits": int(hits[~dev].sum())})
    reports.sort(key=lambda r: -r["hits"])
    baseline = next(r for r in reports if (r["k1"], r["b"], r["last_weight"], r["title_weight"]) == (1.5, .75, 0, 0))
    data = {"scope": "public_retrieval_only_already_inspected", "count": len(items),
            "dev_count": int(dev.sum()), "other_count": int((~dev).sum()), "baseline": baseline,
            "public_sha256": hashlib.sha256((ROOT / "public/tutor.jsonl").read_bytes()).hexdigest(),
            "grid": reports}
    dest = ROOT / "experiments/.cache/jh_retrieval_grid"
    dest.mkdir(exist_ok=True)
    (dest / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"count": len(items), "baseline": baseline, "best": reports[:10]}))


if __name__ == "__main__":
    run()
