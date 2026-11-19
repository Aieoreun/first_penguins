"""Screen section confidence using public fitting, grouped OOF, and KB diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from core.kb import load_chunks
from core.search import Searcher
from pipelines.kimjaehak.learned_sections import train_section_classifier
from jh_section_ranker_cv import groups_for

THRESHOLDS=(.5,.6,.7,.8,.85,.9,.925,.95,.96,.97,.98,.99,.995)


def metrics(rows, threshold):
    selected=[r['predicted'] if r['confidence']>=threshold else r['base'] for r in rows]
    return {'count':len(rows),'hits':sum(p in r['gold'] for p,r in zip(selected,rows)),
            'overrides':sum(p!=r['base'] for p,r in zip(selected,rows))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dataset',action='append',default=[],help='name=local derived_inputs.jsonl')
    args=ap.parse_args()
    dest=args.output.resolve()
    if not dest.is_relative_to(ROOT/'experiments/.cache') or dest.exists():
        ap.error('Use a new local cache directory')
    sources={'public':ROOT/'public/tutor.jsonl'}
    for value in args.dataset:
        name,path=value.split('=',1)
        path=Path(path).resolve()
        if name in sources or not path.is_relative_to(ROOT/'experiments/.cache'):
            ap.error('Invalid diagnostic path/name')
        meta=json.loads((path.parent/'derivation.json').read_text())
        if (meta['dataset_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()
                or meta['scope'] not in {'provided_kb_fact_diagnostic','public_paraphrase_full_diagnostic'}):
            ap.error('Unexpected diagnostic provenance')
        sources[name]=path
    searcher=Searcher(load_chunks(ROOT/'kb'))
    vec,model=train_section_classifier(1000)
    features={}
    public_rows=None
    for name,path in sources.items():
        rows=[r for r in map(json.loads,path.read_text().splitlines()) if len(r['labels'].get('evidence') or [])==1]
        texts=[next(m['content'] for m in reversed(r['messages']) if m['role']=='user') for r in rows]
        probabilities=model.predict_proba(vec.transform(texts))
        features[name]=[]
        for row,ps in zip(rows,probabilities):
            base=searcher.search('\n'.join(m['content'] for m in row['messages']),bm25_top=1,embedding_top=0)[0].section_id
            j=ps.argmax()
            features[name].append({'id':row['id'],'base':base,'predicted':str(model.classes_[j]),
                                   'confidence':float(ps[j]),'gold':row['labels']['evidence']})
        if name=='public':
            public_rows=rows
    rows=public_rows
    texts=np.array([next(m['content'] for m in reversed(r['messages']) if m['role']=='user') for r in rows])
    labels=np.array([r['labels']['evidence'][0] for r in rows])
    groups=groups_for(rows)
    by_id={r['id']:r for r in features['public']}
    oof=[]
    with threadpool_limits(limits=1):
        for train,test in GroupKFold(5,shuffle=True,random_state=20261116).split(rows,groups=groups):
            fold_vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=1,max_features=60000,sublinear_tf=True)
            fold_model=LogisticRegression(C=1000,max_iter=500,class_weight='balanced')
            fold_model.fit(fold_vec.fit_transform(texts[train]),labels[train])
            for i,ps in zip(test,fold_model.predict_proba(fold_vec.transform(texts[test]))):
                j=ps.argmax()
                oof.append({**by_id[rows[i]['id']],'predicted':str(fold_model.classes_[j]),'confidence':float(ps[j])})
    reports=[{'threshold':t,'metrics':{name:metrics(data,t) for name,data in features.items()}} for t in THRESHOLDS]
    result={'scope':'public_fitted_section_gate_development_grid_not_holdout','features':features,'reports':reports,
            'dataset_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()},
            'oof':{'groups':len(set(groups)),'features':oof,'reports':[{'threshold':t,**metrics(oof,t)} for t in THRESHOLDS],
                   'scope':'grouped_oof_on_previously_used_public_development_questions'}}
    dest.mkdir()
    (dest/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for report in reports:
        print(report['threshold'],{k:v['hits'] for k,v in report['metrics'].items()})


if __name__=='__main__':main()
