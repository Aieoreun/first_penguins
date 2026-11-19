"""Grouped OOF and in-sample diagnostics for public-trained span boundaries."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from scoring import span_f1
from pipelines.kimjaehak.span_tagger import SpanTagger, public_rows, refine_spans


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--regularizations", type=float, nargs="+", default=[10, 100, 1000])
    args = ap.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        ap.error("Only recorded public predictions are supported")
    rows = public_rows()
    recorded = {r["id"]: r["response"]["output"]["unsupported"]
                for r in map(json.loads, source.read_text().splitlines())
                if r["endpoint"] == "guard" and r["response"].get("output") is not None}
    texts = [r["candidate_answer"] for r in rows]
    groups = list(range(len(rows)))
    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i
    x = TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(texts)
    similarity = (x @ x.T).toarray()
    for i in range(len(rows)):
        for j in range(i):
            if similarity[i, j] >= .75:
                groups[find(i)] = find(j)
    groups = [find(i) for i in range(len(rows))]
    labels = [bool(r["labels"]["unsupported"]) for r in rows]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261116).split(texts, labels, groups))
    def score(predictions, threshold):
        raw, combined, full_gate, used, covered = [], [], [], 0, 0
        for row, (pred, confidence) in zip(rows, predictions):
            gold = row["labels"]["unsupported"]
            original = recorded[row["id"]]
            selected, changed = refine_spans(row["candidate_answer"], original, pred, confidence, threshold)
            raw.append(span_f1(pred, gold)); combined.append(span_f1(selected, gold)); used += changed
            covered += confidence >= threshold
            full_gate.append(span_f1(pred if confidence >= threshold else original, gold))
        return {"raw_span_f1":sum(raw)/len(raw), "refined_span_f1":sum(combined)/len(combined), "refinements":used,
                "confidence_gate_span_f1":sum(full_gate)/len(full_gate), "confidence_gate_coverage":covered}
    reports = []
    with threadpool_limits(limits=1):
        for c in args.regularizations:
            predictions = [None] * len(rows)
            for train, test in folds:
                assert not {groups[i] for i in train}.intersection(groups[i] for i in test)
                model = SpanTagger([rows[i] for i in train], c)
                for i in test:
                    predictions[i] = model.predict(texts[i])
            model = SpanTagger(rows, c)
            fitted = [model.predict(text) for text in texts]
            report = {"C":c,"gates":{str(t):{"oof":score(predictions,t),"in_sample":score(fitted,t)}
                                     for t in (.8,.9,.95,.99)}}
            reports.append(report); print(json.dumps(report),flush=True)
    data = {"scope":"public_grouped_oof_span_diagnostic","count":len(rows),"groups":len(set(groups)),
            "baseline_span_f1":sum(span_f1(recorded[r["id"]],r["labels"]["unsupported"]) for r in rows)/len(rows),
            "source":str(source.relative_to(ROOT)),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "public_sha256":hashlib.sha256((ROOT/"public/guard.jsonl").read_bytes()).hexdigest(),
            "analysis_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "no_new_llm_calls":True,"reports":reports}
    args.output.mkdir(exist_ok=False)
    (args.output/"diagnostic.json").write_text(json.dumps(data,indent=2)+"\n")


if __name__ == "__main__":
    main()
