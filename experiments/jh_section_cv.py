"""Audit a scratch section classifier using only public evidence labels.

Report fitted-public and grouped out-of-fold results separately. A prediction
can replace BM25 only when the section is already in its retrieved candidates.
"""
import argparse
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


def run(kb_weight=0, regularizations=(10, 100, 1000, 5000)):
    source = ROOT / "public/tutor.jsonl"
    rows = [r for r in map(json.loads, source.read_text().splitlines()) if len(r["labels"].get("evidence") or []) == 1]
    texts = [next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user") for r in rows]
    gold = np.array([r["labels"]["evidence"][0] for r in rows])
    searcher = Searcher(load_chunks(ROOT / "kb"))
    extra_texts, extra_labels, augmentation = [], [], None
    if kb_weight:
        data = ROOT / "pipelines/kimjaehak/data"
        raw = (data / "kb_questions.jsonl").read_bytes()
        metadata = json.loads((data / "kb_questions_manifest.json").read_text())
        sections = {sid:searcher.section_text(sid) for sid in searcher.section_ids()}
        digest = hashlib.sha256(json.dumps(sections,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if metadata["scope"] != "provided_llm_kb_only_questions" or metadata["source_sha256"] != digest or metadata["questions_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("KB augmentation provenance mismatch")
        for row in map(json.loads, raw.decode().splitlines()):
            if row["error"] is None:
                if row["section"] not in sections:
                    raise ValueError("Unknown source section")
                extra_texts.extend(row["questions"])
                extra_labels.extend([row["section"]] * len(row["questions"]))
        augmentation = {"weight": kb_weight, "examples": len(extra_texts), "provenance": metadata}
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
    for c in regularizations:
        def fit(train):
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                                         max_features=60000, sublinear_tf=True)
            model = LogisticRegression(C=c, max_iter=500, class_weight="balanced")
            fit_texts = [texts[i] for i in train] + extra_texts
            fit_labels = list(gold[train]) + extra_labels
            model.fit(vectorizer.fit_transform(fit_texts), fit_labels,
                      sample_weight=[1.] * len(train) + [kb_weight] * len(extra_texts))
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
        for limit in (3, 10, None):
            for threshold in (.9, .95, .98, .995):
                def measure(pred, conf):
                    mask = np.array([conf[i] >= threshold and pred[i] in (searcher.section_ids() if limit is None else candidates[i][:limit]) for i in range(len(rows))])
                    return {"coverage":int(mask.sum()),"selected_hits":int((pred[mask]==gold[mask]).sum()),
                            "combined_hits":int((np.where(mask,pred,baseline)==gold).sum())}
                report["gates"][f"{'all' if limit is None else 'top'+str(limit)}@{threshold}"] = {"oof":measure(predicted,confidence),
                    "training":measure(fitted,probability.max(axis=1))}
        reports.append(report)
        print(json.dumps(report),flush=True)
    return {"scope":"public_section_classifier_grouped_oof", "count":len(rows), "groups":len(set(groups)),
            "baseline_hits":int((baseline==gold).sum()),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "augmentation": augmentation, "reports":reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-weight", type=float, default=0)
    parser.add_argument("--regularizations", type=float, nargs="+", default=[10, 100, 1000, 5000])
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/.cache/jh_section_cv")
    args = parser.parse_args()
    if not 0 <= args.kb_weight <= 1 or not all(0 < c <= 1e6 for c in args.regularizations):
        parser.error("Invalid weights or regularization")
    with threadpool_limits(limits=1):
        report = run(args.kb_weight, args.regularizations)
    dest = args.output
    dest.mkdir(exist_ok=True)
    (dest / "diagnostic.json").write_text(json.dumps(report, indent=2))
