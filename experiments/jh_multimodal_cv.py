"""Compare scratch classifiers on fixed public grouped folds; no new API calls."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from pipelines.kimjaehak.input_examples import PolicyExamples

BLOCKED = {"harmful_content","academic_dishonesty","jailbreak_injection","off_topic"}


def main():
    examples = PolicyExamples(True)
    rows,texts,dense = examples.rows,examples.texts,examples.vectors
    y = np.array([r["labels"]["category"] for r in rows])
    groups = list(range(len(rows)))
    def find(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i
    near = TfidfVectorizer(analyzer="char",ngram_range=(3,5)).fit_transform(texts)
    sim = (near @ near.T).toarray()
    for i in range(len(rows)):
        for j in range(i):
            if sim[i,j] >= .82:
                groups[find(i)] = find(j)
    groups = [find(i) for i in range(len(rows))]
    folds = list(StratifiedGroupKFold(5,shuffle=True,random_state=20261116).split(texts,y,groups))
    source = ROOT / "experiments/.cache/jh_gate_fallback_round2_current/jh_adaptive_1/responses.jsonl"
    records = [json.loads(s) for s in source.read_text().splitlines()]
    by_id = {r["id"]:r["response"]["input"]["category"] for r in records}
    assert len(records)==400 and set(by_id)=={r["id"] for r in rows}
    fallback = np.array([by_id[r["id"]] for r in rows])
    configurations = [("char1000","char",0,1000),("char3000","char",0,3000),("dense1000","dense",0,1000),
                      ("mix05","mix",.5,1000),("mix10","mix",1,1000),("mix20","mix",2,1000),
                      ("dense_rbf1","rbf",1,10),("dense_rbf5","rbf",5,10)]
    predicted = {name:np.empty(len(rows),dtype=object) for name,*_ in configurations}
    confidence = {name:np.zeros(len(rows)) for name in predicted}
    with threadpool_limits(limits=1):
        for train,test in folds:
            assert not set(groups[i] for i in train)&set(groups[i] for i in test)
            vec = TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
            char_train = vec.fit_transform([texts[i] for i in train])
            char_test = vec.transform([texts[i] for i in test])
            for name,kind,weight,c in configurations:
                if kind=="char":
                    a,b = char_train,char_test
                elif kind=="mix":
                    a = normalize(hstack([char_train,csr_matrix(weight*dense[train])]).tocsr())
                    b = normalize(hstack([char_test,csr_matrix(weight*dense[test])]).tocsr())
                else:
                    a,b = dense[train],dense[test]
                model = (SVC(C=c,gamma=weight,probability=True,class_weight="balanced",random_state=20261116)
                         if kind=="rbf" else LogisticRegression(C=c,max_iter=1000,class_weight="balanced"))
                model.fit(a,y[train])
                p = model.predict_proba(b)
                predicted[name][test],confidence[name][test] = model.classes_[p.argmax(axis=1)],p.max(axis=1)
    def metrics(pred):
        gb,pb = np.array([x in BLOCKED for x in y]),np.array([x in BLOCKED for x in pred])
        return {"category_non_normal":float((pred[y!="normal"]==y[y!="normal"]).mean()),
                "block_f1":float(f1_score(gb,pb)),"false_block":int((~gb&pb).sum()),"missed_block":int((gb&~pb).sum())}
    reports = []
    for name,*_ in configurations:
        pred,conf = predicted[name],confidence[name]
        report = {"name":name,"oof_accuracy":float((pred==y).mean()),"gates":{}}
        for t in (.95,.98,.995):
            mask = (conf>=t)&(pred!="harmful_content")
            report["gates"][str(t)] = {"coverage":int(mask.sum()),"precision":float((pred[mask]==y[mask]).mean()) if mask.any() else None,
                                       "combined":metrics(np.where(mask,pred,fallback))}
        reports.append(report)
    report = {"scope":"public_grouped_oof_multimodal_screen_not_holdout","groups":len(set(groups)),
              "baseline_fallback":metrics(fallback),"reports":reports,
              "fallback_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"excludes_harmful_fast_subtype_route":True}
    target = ROOT / "experiments/.cache/jh_embedding_cv_r2/multimodal_cv.json"
    if target.exists():
        raise ValueError("Refusing to overwrite a completed diagnostic")
    target.write_text(json.dumps(report,indent=2))
    for r in reports:
        print(json.dumps(r),flush=True)


if __name__ == "__main__":
    main()
