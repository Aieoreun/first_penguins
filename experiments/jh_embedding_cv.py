"""Grouped OOF category diagnostic using organizer-provided embeddings only."""
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from core.llm import LLM
from core.search import normalize


async def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    out = ROOT / "experiments/.cache/jh_embedding_cv_r2"
    out.mkdir(exist_ok=False)
    paths = [ROOT / "public" / f"{e}.jsonl" for e in ("tutor", "guard")]
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    texts = [next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user") for r in rows]
    y = np.array([r["labels"]["category"] for r in rows])
    llm = LLM()
    if not llm.can_embed:
        raise RuntimeError("Configure organizer embeddings")
    vectors = []
    for start in range(0,len(texts),32):
        batch = await llm.embed(texts[start:start+32],timeout=60)
        if len(batch) != len(texts[start:start+32]):
            raise ValueError("Incomplete embeddings")
        vectors.extend(normalize(v) for v in batch)
    x = np.array(vectors)
    if x.shape != (len(rows),1536) or not np.isfinite(x).all():
        raise ValueError("Invalid vectors")
    metadata = {"model":llm.embedding_model,"public_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "scope":"public_category_embeddings", "ids":[r["id"] for r in rows],
                "text_mode":"last_user", "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    np.savez_compressed(out / "public_vectors.npz", vectors=x)
    (out / "manifest.json").write_text(json.dumps(metadata,indent=2))
    groups = list(range(len(rows)))
    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i
    near = TfidfVectorizer(analyzer="char",ngram_range=(3,5)).fit_transform(texts)
    similarity = (near @ near.T).toarray()
    for i in range(len(rows)):
        for j in range(i):
            if similarity[i,j] >= .82:
                groups[find(i)] = find(j)
    groups = [find(i) for i in range(len(rows))]
    folds = list(StratifiedGroupKFold(5,shuffle=True,random_state=20261116).split(x,y,groups))
    reports = []
    with threadpool_limits(limits=1):
        for c in (10,100,1000):
            prediction = np.empty(len(rows),dtype=object)
            confidence = np.zeros(len(rows))
            for train,test in folds:
                model = LogisticRegression(C=c,max_iter=1000,class_weight="balanced").fit(x[train],y[train])
                probability = model.predict_proba(x[test])
                prediction[test] = model.classes_[probability.argmax(axis=1)]
                confidence[test] = probability.max(axis=1)
            gates = {}
            for t in (.9,.95,.98):
                mask = (confidence >= t) & (prediction != "harmful_content")
                gates[str(t)] = {"coverage":int(mask.sum()),"accuracy":float((prediction[mask]==y[mask]).mean()) if mask.any() else None}
            report = {"C":c,"oof_accuracy":float((prediction==y).mean()),"gates":gates,
                      "errors":[{"id":rows[i]["id"],"gold":str(y[i]),"predicted":str(prediction[i]),"confidence":float(confidence[i])} for i in range(len(rows)) if y[i] != prediction[i]]}
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k != "errors"}),flush=True)
    (out / "result.json").write_text(json.dumps({"scope":"public_grouped_oof_embedding_diagnostic_not_holdout","groups":len(set(groups)),"reports":reports},indent=2))


if __name__ == "__main__":
    asyncio.run(main())
