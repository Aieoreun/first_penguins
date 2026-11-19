"""Reproduce the rejected public-only classifier experiment; no model is deployed.

Near-duplicate last requests are grouped before 5-fold validation. Each fold fits
its own character TF-IDF vocabulary and logistic regression from scratch.
Install requirements-jh-research.txt; the production server does not need it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {"harmful_content", "academic_dishonesty", "jailbreak_injection", "off_topic"}


def evaluate():
    paths = [ROOT / "public" / f"{e}.jsonl" for e in ("tutor", "guard")]
    rows = [json.loads(s) for path in paths for s in path.read_text().splitlines()]
    last = [next(m["content"] for m in reversed(p["messages"]) if m["role"] == "user") for p in rows]
    history = ["\n".join(m["content"] for m in p["messages"][:-1]) for p in rows]
    y = np.array([p["labels"]["category"] for p in rows])
    size = len(rows)
    features = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1).fit_transform(last)
    similarity = (features @ features.T).toarray()
    parent = list(range(size))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(size):
        for j in range(i):
            if similarity[i, j] >= 0.82:
                parent[find(i)] = find(j)
    groups = np.array([find(i) for i in range(size)])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261116).split(last, y, groups))
    reports = []
    for context in (False, True):
        for regularization in (0.3, 1.0, 3.0, 10.0):
            predicted, confidence = np.empty(size, dtype=object), np.zeros(size)
            for train, test in folds:
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                             max_features=60000, sublinear_tf=True)
                x = vectorizer.fit_transform([last[i] for i in train])
                z = vectorizer.transform([last[i] for i in test])
                if context:
                    history_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                        min_df=2, max_features=30000, sublinear_tf=True)
                    x = hstack([x, history_vectorizer.fit_transform([history[i] for i in train]) * 0.5])
                    z = hstack([z, history_vectorizer.transform([history[i] for i in test]) * 0.5])
                model = LogisticRegression(C=regularization, max_iter=500, class_weight="balanced").fit(x, y[train])
                probability = model.predict_proba(z)
                predicted[test] = model.classes_[probability.argmax(axis=1)]
                confidence[test] = probability.max(axis=1)
            report = {"context": context, "C": regularization,
                "accuracy": float(accuracy_score(y, predicted)),
                "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
                "block_f1": float(f1_score([s in BLOCKED for s in y], [s in BLOCKED for s in predicted])),
                "thresholds": {}}
            for threshold in (0.5, 0.7, 0.8, 0.9):
                mask = confidence >= threshold
                report["thresholds"][str(threshold)] = {"coverage": int(mask.sum()),
                    "accuracy": float(accuracy_score(y[mask], predicted[mask])) if mask.any() else None}
            reports.append(report)
            print(json.dumps(report), flush=True)
    return {"items": size, "groups": len(set(groups)), "group_cosine_threshold": 0.82,
            "fold_seed": 20261116, "sklearn": sklearn.__version__,
            "public_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/.cache/jh_ml_cv/reproduced.json")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
