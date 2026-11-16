"""Public-group OOF and frozen KB diagnostics for generic section pair features."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from core.kb import load_chunks
from core.search import Searcher
from pipelines.kimjaehak.section_ranker import SectionFeatures


def groups_for(rows):
    last=[next(m['content'] for m in reversed(r['messages']) if m['role']=='user') for r in rows]
    matrix=TfidfVectorizer(analyzer='char',ngram_range=(3,5)).fit_transform(last)
    similarity=(matrix@matrix.T).toarray()
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i in range(len(rows)):
        for j in range(i):
            if similarity[i,j]>=.82:parent[find(i)]=find(j)
    return np.array([find(i) for i in range(len(rows))])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    dest=args.output.resolve()
    if not dest.is_relative_to(ROOT/'experiments/.cache') or dest.exists():
        ap.error('Use a new local cache directory')
    dest.mkdir()
    searcher=Searcher(load_chunks(ROOT/'kb'))
    features=SectionFeatures(searcher)
    rows=[r for r in map(json.loads,(ROOT/'public/tutor.jsonl').read_text().splitlines())
          if len(r['labels'].get('evidence') or [])==1]
    groups=groups_for(rows)
    x=np.array([features.transform(r['messages']) for r in rows])
    y=np.array([[int(sid in r['labels']['evidence']) for sid in features.ids] for r in rows])
    def model(name):
        if name.startswith('logistic'):
            return make_pipeline(StandardScaler(), LogisticRegression(C=float(name.split('_')[1]),
                class_weight='balanced', max_iter=1000))
        return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=int(name.split('_')[1]),
            min_samples_leaf=25,l2_regularization=5,class_weight='balanced',random_state=20261116)
    baseline=[searcher.search('\n'.join(m['content'] for m in r['messages']))[0].section_id for r in rows]
    def report(data,scores,base):
        predictions=np.argmax(scores,axis=1)
        correct=[features.ids[p] in r['labels']['evidence'] for p,r in zip(predictions,data)]
        return {'count':len(data),'top1':sum(correct),
                'top3':sum(bool(set(r['labels']['evidence']) & {features.ids[i] for i in np.argsort(-s)[:3]}) for r,s in zip(data,scores)),
                'baseline_top1':sum(b in r['labels']['evidence'] for b,r in zip(base,data)),
                'predictions':[{'id':r['id'],'section':features.ids[p],'correct':bool(c)} for r,p,c in zip(data,predictions,correct)]}
    external={}
    for name,folder in [('kb55','jh_kb_probe_r2_base_off'),('kb57','jh_kb_probe_r2_split_kb57_1_jh_guard_fitted_contextfast')]:
        source=ROOT/'experiments/.cache'/folder/'derived_inputs.jsonl'
        provenance=json.loads((source.parent/'derivation.json').read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest()!=provenance['dataset_sha256']:
            raise ValueError('Dataset changed')
        data=list(map(json.loads,source.read_text().splitlines()))
        external[name]=(data,np.array([features.transform(r['messages']) for r in data]),
                        [searcher.search('\n'.join(m['content'] for m in r['messages']))[0].section_id for r in data],source)
    reports={}
    with threadpool_limits(limits=1):
        for name in ('logistic_0.01','logistic_0.1','logistic_1','trees_3','trees_7'):
            scores=np.zeros(y.shape)
            for train,test in GroupKFold(5,shuffle=True,random_state=20261116).split(rows,groups=groups):
                estimator=model(name)
                estimator.fit(x[train].reshape(-1,x.shape[-1]),y[train].ravel())
                scores[test]=estimator.predict_proba(x[test].reshape(-1,x.shape[-1]))[:,1].reshape(len(test),-1)
            metrics={'public_grouped_oof':report(rows,scores,baseline)}
            estimator=model(name)
            estimator.fit(x.reshape(-1,x.shape[-1]),y.ravel())
            for dataset,(data,xx,base,source) in external.items():
                pred=estimator.predict_proba(xx.reshape(-1,x.shape[-1]))[:,1].reshape(len(data),-1)
                metrics[dataset]=report(data,pred,base)
                metrics[dataset]['dataset_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
            reports[name]=metrics
            print(json.dumps({name:{k:{f:v for f,v in m.items() if f!='predictions'} for k,m in metrics.items()}}),flush=True)
    result={'scope':'section_ranking_public_grouped_oof_and_development_kb','groups':len(set(groups)),
            'features':x.shape[-1],'source_sha256':hashlib.sha256((ROOT/'public/tutor.jsonl').read_bytes()).hexdigest(),
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'external_data_used_for_training':False,'reports':reports}
    (dest/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
