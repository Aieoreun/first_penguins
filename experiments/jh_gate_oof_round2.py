"""Screen non-harmful category gates against fresh ungated PUBLIC decisions.

OOF models never see their validation groups. Prompts have seen the public set;
this is a development diagnostic, not a holdout or a full production score.
"""
import argparse
import hashlib
import json
from pathlib import Path

from jh_compare import ROOT, evaluator_hashes
from jh_input_fallback import BLOCKED
from pipelines.kimjaehak.input_examples import PolicyExamples

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT / "experiments/.cache"):
        parser.error("Use the local public fallback diagnostic")
    meta = json.loads((source / "manifest.json").read_text())
    if meta.get("scope") != "public_ungated_input_fallback_for_oof_diagnostic" or meta["evaluator_hashes"] != evaluator_hashes():
        parser.error("Fallback provenance differs")
    examples = PolicyExamples(True)
    rows, texts = examples.rows, examples.texts
    y = np.array([r["labels"]["category"] for r in rows])
    x = examples.vectors
    records = [json.loads(s) for s in (source / "responses.jsonl").read_text().splitlines()]
    responses = {r["id"]:r["response"] for r in records}
    if len(records) != 400 or set(responses) != {r["id"] for r in rows}:
        parser.error("Incomplete fallback data")
    fallback = np.array([responses[r["id"]]["input"]["category"] for r in rows])
    groups = list(range(len(rows)))
    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i
    near = TfidfVectorizer(analyzer="char",ngram_range=(3,5)).fit_transform(texts)
    similarities = (near @ near.T).toarray()
    for i in range(len(rows)):
        for j in range(i):
            if similarities[i,j] >= .82:
                groups[find(i)] = find(j)
    groups = [find(i) for i in range(len(rows))]
    folds = list(StratifiedGroupKFold(5,shuffle=True,random_state=20261116).split(texts,y,groups))
    predictions = {c:np.empty(len(rows),dtype=object) for c in (1000,3000)}
    confidence = {c:np.zeros(len(rows)) for c in predictions}
    dp, dc = np.empty(len(rows),dtype=object), np.zeros(len(rows))
    with threadpool_limits(limits=1):
        for train,test in folds:
            vec = TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
            a,b = vec.fit_transform([texts[i] for i in train]), vec.transform([texts[i] for i in test])
            for c in predictions:
                model = LogisticRegression(C=c,max_iter=500,class_weight="balanced").fit(a,y[train])
                p = model.predict_proba(b)
                predictions[c][test],confidence[c][test] = model.classes_[p.argmax(axis=1)],p.max(axis=1)
            dense = LogisticRegression(C=1000,max_iter=1000,class_weight="balanced").fit(x[train],y[train])
            p = dense.predict_proba(x[test])
            dp[test],dc[test] = dense.classes_[p.argmax(axis=1)],p.max(axis=1)
    guard = np.array([r["endpoint"]=="guard" for r in rows])
    def metrics(pred):
        result = {}
        for name,mask in (("all",np.ones(len(rows),dtype=bool)),("original_guard",guard),("original_tutor",~guard)):
            gold,guess = y[mask], pred[mask]
            gb,pb = np.array([x in BLOCKED for x in gold]),np.array([x in BLOCKED for x in guess])
            result[name] = {"category_non_normal":float((gold[gold!="normal"]==guess[gold!="normal"]).mean()),
                            "block_f1":float(f1_score(gb,pb)),"false_block":int((~gb&pb).sum()),"missed_block":int((gb&~pb).sum())}
        return result
    configurations = [("current95",.95,3000,None),("threshold98",.98,3000,None),
                      ("threshold995",.995,3000,None),("c1000_threshold995",.995,1000,None),
                      ("semantic_agreement",.95,3000,0),("semantic95_agreement",.95,3000,.95)]
    reports = []
    for name,threshold,guard_c,semantic in configurations:
        pred = np.where(guard,predictions[guard_c],predictions[1000])
        conf = np.where(guard,confidence[guard_c],confidence[1000])
        mask = (conf >= threshold) & (pred != "harmful_content")
        if semantic is not None:
            mask &= (pred==dp)&(dc>=semantic)
        combined = np.where(mask,pred,fallback)
        reports.append({"name":name,"coverage":int(mask.sum()),"selected_accuracy":float((pred[mask]==y[mask]).mean()),
                        "changed":int((combined!=fallback).sum()),"metrics":metrics(combined)})
    result = {"scope":"public_grouped_oof_nonharmful_gate_screen_not_holdout","groups":len(set(groups)),
              "baseline":metrics(fallback),"reports":reports,"source":str(source.relative_to(ROOT)),
              "source_sha256":hashlib.sha256((source / "responses.jsonl").read_bytes()).hexdigest(),
              "excludes_harmful_fast_subtype_route":True}
    destination = source.parent / "gate_oof.json"
    if destination.exists():
        parser.error("Refusing to overwrite completed diagnostic")
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
