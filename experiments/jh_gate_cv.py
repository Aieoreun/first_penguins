"""Test a confidence-gated scratch classifier against recorded LLM predictions.

Only public classification labels are used. This produces offline diagnostics,
not model artifacts. Fold vocabularies and models never see their test rows;
the public set has already been inspected during prompt development.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

BLOCKED = {"harmful_content", "academic_dishonesty", "jailbreak_injection", "off_topic"}


def run(source=None, exclude_harmful=False):
    paths = [ROOT / "public" / f"{e}.jsonl" for e in ("tutor", "guard")]
    rows = [p for path in paths for p in map(json.loads, path.read_text().splitlines())]
    texts = [next(m["content"] for m in reversed(p["messages"]) if m["role"] == "user") for p in rows]
    y = np.array([p["labels"]["category"] for p in rows])
    size = len(rows)
    groups = list(range(size))
    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i
    near = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(texts)
    similarity = (near @ near.T).toarray()
    for i in range(size):
        for j in range(i):
            if similarity[i, j] >= .82:
                groups[find(i)] = find(j)
    groups = [find(i) for i in range(size)]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261116).split(texts, y, groups))
    source = source or ROOT / "experiments/.cache/jh_full_voted_true64/jh_voted_1/responses.jsonl"
    recorded = {r["id"]: r["response"] for r in map(json.loads, source.read_text().splitlines())}
    guard = np.array([p["endpoint"] == "guard" for p in rows])
    baseline = np.array([recorded[p["id"]]["input"]["category"] if p["endpoint"] == "guard" else "" for p in rows])
    def metrics(pred):
        gold, pred = y[guard], pred[guard]
        return {"category_accuracy_non_normal": float(accuracy_score(gold[gold != "normal"], pred[gold != "normal"])),
                "block_f1": float(f1_score([v in BLOCKED for v in gold], [v in BLOCKED for v in pred]))}
    reports = []
    for c in (10, 100, 300, 1000):
        predicted, confidence = np.empty(size, dtype=object), np.zeros(size)
        def fit(indices):
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                         max_features=60000, sublinear_tf=True)
            x = vectorizer.fit_transform([texts[i] for i in indices])
            model = LogisticRegression(C=c, max_iter=500, class_weight="balanced").fit(x, y[indices])
            return vectorizer, model
        for train, test in folds:
            vectorizer, model = fit(train)
            probability = model.predict_proba(vectorizer.transform([texts[i] for i in test]))
            predicted[test] = model.classes_[probability.argmax(axis=1)]
            confidence[test] = probability.max(axis=1)
        vectorizer, model = fit(np.arange(size))
        train_probability = model.predict_proba(vectorizer.transform(texts))
        train_prediction = model.classes_[train_probability.argmax(axis=1)]
        report = {"C": c, "out_of_fold_accuracy": float(accuracy_score(y, predicted)),
                  "training_accuracy": float(accuracy_score(y, train_prediction)), "gates": {}}
        for threshold in (.9, .95, .98, .995):
            mask = confidence >= threshold
            if exclude_harmful:
                mask &= predicted != "harmful_content"
            combined = np.where(mask, predicted, baseline)
            train_mask = train_probability.max(axis=1) >= threshold
            if exclude_harmful:
                train_mask &= train_prediction != "harmful_content"
            report["gates"][str(threshold)] = {
                "oof_coverage": int(mask.sum()), "oof_selected_accuracy": float(accuracy_score(y[mask], predicted[mask])) if mask.any() else None,
                "guard_replacements": int((guard & mask & (predicted != baseline)).sum()),
                "combined_guard_input": metrics(combined), "training_coverage": int(train_mask.sum()),
                "training_selected_accuracy": float(accuracy_score(y[train_mask], train_prediction[train_mask])) if train_mask.any() else None}
        reports.append(report)
        print(json.dumps(report), flush=True)
    return {"scope": "public_grouped_oof_gate_diagnostic", "groups": len(set(groups)),
            "exclude_harmful": exclude_harmful,
            "baseline": metrics(baseline), "source": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "public_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, "reports": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--exclude-harmful", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/.cache/jh_gate_cv")
    args = parser.parse_args()
    source = args.source.resolve() if args.source else None
    if source and not source.is_relative_to(ROOT / "experiments/.cache"):
        parser.error("Only local public recordings are supported")
    with threadpool_limits(limits=1):
        data = run(source, args.exclude_harmful)
    dest = args.output
    dest.mkdir(exist_ok=True)
    (dest / "result.json").write_text(json.dumps(data, indent=2) + "\n")
