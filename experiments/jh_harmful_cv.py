"""Public-only subtype OOF and optional seen-parent paraphrase diagnostics."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from pipelines.kimjaehak.learned_harmful import harmful_training_rows, train_harmful_classifier


def last(row):
    return next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user")


def metrics(model, vectorizer, rows):
    probability = model.predict_proba(vectorizer.transform([last(row) for row in rows]))
    return model.classes_[probability.argmax(axis=1)], probability.max(axis=1)


def summarize(predicted, confidence, labels):
    return {"count": len(labels), "accuracy": float(np.mean(predicted == labels)),
            "gates": {str(t): {"coverage": int(sum(confidence >= t)),
                               "hits": int(sum((confidence >= t) & (predicted == labels)))}
                      for t in (.95, .98, .995)}}


def run(derived):
    rows = harmful_training_rows()
    texts = [last(row) for row in rows]
    labels = np.array([row["labels"]["action"] for row in rows])
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    features = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(texts)
    similarity = (features @ features.T).toarray()
    for i in range(len(rows)):
        for j in range(i):
            if similarity[i, j] >= .82:
                parents[find(i)] = find(j)
    groups = [find(i) for i in range(len(rows))]
    predicted, confidence = np.empty(len(rows), dtype=object), np.zeros(len(rows))
    for train, test in StratifiedGroupKFold(4, shuffle=True, random_state=20261116).split(texts, labels, groups):
        assert not set(groups[i] for i in train) & set(groups[i] for i in test)
        vectorizer, model = train_harmful_classifier(1000, [rows[i] for i in train])
        predicted[test], confidence[test] = metrics(model, vectorizer, [rows[i] for i in test])
    report = {"scope": "public_harmful_subtype_diagnostic", "C": 1000, "groups": len(set(groups)),
              "public_tutor_sha256": hashlib.sha256((ROOT / "public/tutor.jsonl").read_bytes()).hexdigest(),
              "grouped_oof": summarize(predicted, confidence, labels), "derived": []}
    vectorizer, model = train_harmful_classifier(1000)
    pred, conf = metrics(model, vectorizer, rows)
    report["training"] = summarize(pred, conf, labels)
    sources = {row["id"]: row for row in rows}
    for path in derived:
        path = path.resolve()
        if not path.is_relative_to(ROOT / "experiments/.cache"):
            raise ValueError("Use a recorded public-derived dataset")
        provenance = json.loads((path.parent / "derivation.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["dataset_sha256"]:
            raise ValueError("Derived data changed")
        if not provenance["scope"].startswith("public_paraphrase_"):
            raise ValueError("Unsupported source")
        selected = [row for row in map(json.loads, path.read_text().splitlines()) if row["source_id"] in sources]
        expected = np.array([sources[row["source_id"]]["labels"]["action"] for row in selected])
        pred, conf = metrics(model, vectorizer, selected)
        report["derived"].append({"dataset": str(path.relative_to(ROOT)), "provenance": provenance,
                                  **summarize(pred, conf, expected)})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    with threadpool_limits(limits=1):
        report = run(args.derived)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
