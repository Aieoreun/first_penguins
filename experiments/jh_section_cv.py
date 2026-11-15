"""Audit a scratch section classifier using only public evidence labels.

Report fitted-public and grouped out-of-fold results separately. A prediction
can replace BM25 only when the section is already in its retrieved candidates.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from core.kb import load_chunks
from core.search import Searcher


def run():
    source = ROOT / "public/tutor.jsonl"
    rows = [r for r in map(json.loads, source.read_text().splitlines()) if len(r["labels"].get("evidence") or []) == 1]
    texts = [next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user") for r in rows]
    gold = np.array([r["labels"]["evidence"][0] for r in rows])
    searcher = Searcher(load_chunks(ROOT / "kb"))
    candidates = [list(dict.fromkeys(c.section_id for c in searcher.search("\n".join(m["content"] for m in r["messages"])))) for r in rows]
    baseline = np.array([c[0] if c else "" for c in candidates])
    near = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(texts)
    similarity = (near @ near.T).toarray()
    groups = list(range(len(rows)))
    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i
    for i in range(len(rows)):
        for j in range(i):
            if similarity[i,j] >= .82:
                groups[find(i)] = find(j)
    groups = [find(i) for i in range(len(rows))]
    folds = list(GroupKFold(5, shuffle=True, random_state=20261116).split(texts, gold, groups))
    reports = []
    for c in (10, 100, 1000, 5000):
        def fit(train):
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                                         max_features=60000, sublinear_tf=True)
            model = LogisticRegression(C=c, max_iter=500, class_weight="balanced")
            model.fit(vectorizer.fit_transform([texts[i] for i in train]), gold[train])
            return vectorizer, model
        predicted, confidence = np.empty(len(rows), dtype=object), np.zeros(len(rows))
        for train, test in folds:
            vectorizer, model = fit(train)
            probability = model.predict_proba(vectorizer.transform([texts[i] for i in test]))
            predicted[test] = model.classes_[probability.argmax(axis=1)]
            confidence[test] = probability.max(axis=1)
        vectorizer, model = fit(np.arange(len(rows)))
        probability = model.predict_proba(vectorizer.transform(texts))
        fitted = model.classes_[probability.argmax(axis=1)]
        report = {"C":c,"oof_hits":int((predicted==gold).sum()),"training_hits":int((fitted==gold).sum()),"gates":{}}
        for limit in (3, 10):
            for threshold in (.9, .95, .98, .995):
                def measure(pred, conf):
                    mask = np.array([conf[i] >= threshold and pred[i] in candidates[i][:limit] for i in range(len(rows))])
                    return {"coverage":int(mask.sum()),"selected_hits":int((pred[mask]==gold[mask]).sum()),
                            "combined_hits":int((np.where(mask,pred,baseline)==gold).sum())}
                report["gates"][f"top{limit}@{threshold}"] = {"oof":measure(predicted,confidence),
                    "training":measure(fitted,probability.max(axis=1))}
        reports.append(report)
        print(json.dumps(report),flush=True)
    return {"scope":"public_section_classifier_grouped_oof", "count":len(rows), "groups":len(set(groups)),
            "baseline_hits":int((baseline==gold).sum()),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "reports":reports}


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        report = run()
    dest = ROOT / "experiments/.cache/jh_section_cv"
    dest.mkdir(exist_ok=True)
    (dest / "diagnostic.json").write_text(json.dumps(report, indent=2))
