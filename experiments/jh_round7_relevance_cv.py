"""Add organizer semantic similarity to existing generic section-pair features.

API zero: only verified cached public/KB-development vectors are read. Public
questions are held out by group; KB55 is never used to fit a ranker.
"""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from core.kb import load_chunks
from core.search import Searcher,load_vectors
from pipelines.kimjaehak.section_ranker import SectionFeatures
from pipelines.kimjaehak.classifier_models import public_data,CategoryModel
from pipelines.kimjaehak.embedding_identity import same_embedding_model
from experiments.jh_section_ranker_cv import groups_for


def main():
    dest=ROOT/'experiments/.cache/jh_r7_relevance1';dest.mkdir()
    old=ROOT/'experiments/.cache/jh_retrieval_r2_initial'
    provenance=json.loads((old/'result.json').read_text())['provenance']
    public,_,last_vectors,model=public_data()
    if not same_embedding_model(model,provenance['embedding_model']):raise ValueError('Embedding identity differs')
    last={r['id']:v for r,v in zip(public,last_vectors)}
    chunks=load_chunks(ROOT/'kb');vectors=None
    for alias in [model,'openai/text-embedding-3-small']:
        vectors=load_vectors(ROOT/'index/kb_embeddings.json',alias,chunks)
        if vectors is not None:break
    matrix=CategoryModel._vectors(vectors,len(chunks))
    searcher=Searcher(chunks,vectors);features=SectionFeatures(searcher)
    public=[r for r in public if r['endpoint']=='tutor' and len(r['labels'].get('evidence') or [])==1]
    source=ROOT/'experiments/.cache/jh_kb_probe_r2_base_off/derived_inputs.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=provenance['kb_probe_sha256']:raise ValueError('KB55 provenance differs')
    kb=[json.loads(s) for s in source.read_text().splitlines()]
    cached={(r['dataset'],r['id']):r for r in map(json.loads,(old/'query_vectors.jsonl').read_text().splitlines())}
    def arrays(rows,dataset):
        oldx=[];newx=[];baseline=[]
        for row in rows:
            query='\n'.join(m['content'] for m in row['messages'])[:6000]
            entry=cached[dataset,row['id']]
            if entry['query_sha256']!=hashlib.sha256(query.encode()).hexdigest():raise ValueError('Query vector mismatch')
            full=np.asarray(entry['vector']);final=last[row['id']] if dataset=='public_tutor' else full
            if dataset!='public_tutor' and len(row['messages'])!=1:raise ValueError('KB full/last mismatch')
            x=features.transform(row['messages']);extra=[];top=[]
            for q in (full,final):
                q=q/np.linalg.norm(q);scores=matrix@q
                maximum=np.array([scores[inds].max() for inds in features.chunk_indices])
                top.append(maximum/max(maximum.max(),1e-12))
                average=np.array([np.sort(scores[inds])[-2:].mean() for inds in features.chunk_indices])
                for values in (maximum,average):
                    ranks=np.array([1+np.count_nonzero(values>v) for v in values])
                    extra.extend([values,values/max(values.max(),1e-12),1/ranks])
            extra.extend([x[:,0]*top[0],x[:,12]*top[1]])
            oldx.append(x);newx.append(np.hstack([x,np.asarray(extra).T]))
            baseline.append(searcher.search(query,bm25_top=1,embedding_top=0)[0].section_id)
        y=np.array([[sid in r['labels']['evidence'] for sid in features.ids] for r in rows])
        return np.asarray(oldx),np.asarray(newx),y,baseline
    with threadpool_limits(limits=1):
        oldx,x,y,baseline=arrays(public,'public_tutor');oldkb,kx,ky,kbaseline=arrays(kb,'kb_development')
    folds=list(GroupKFold(5,shuffle=True,random_state=20261116).split(public,groups=groups_for(public)))
    def estimator(kind):
        if kind.startswith('lr'):
            return make_pipeline(StandardScaler(),LogisticRegression(C=float(kind[2:]),class_weight='balanced',max_iter=1000))
        return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=5,min_samples_leaf=25,
            l2_regularization=5,class_weight='balanced',random_state=20261116)
    def report(rows,scores,base):
        order=np.argsort(-scores,axis=1,kind='stable')
        right=np.array([features.ids[j] in r['labels']['evidence'] for r,j in zip(rows,order[:,0])])
        margin=scores[np.arange(len(rows)),order[:,0]]-scores[np.arange(len(rows)),order[:,1]]
        return {'count':len(rows),'top1':int(right.sum()),'top3':sum(any(features.ids[j] in r['labels']['evidence'] for j in ids[:3]) for r,ids in zip(rows,order)),
            'baseline_top1':sum(s in r['labels']['evidence'] for r,s in zip(rows,base)),
            'margin_precision':{str(t):{'count':int((margin>=t).sum()),'correct':int(right[margin>=t].sum())} for t in [.02,.05,.1]},
            'predictions':[{'id':r['id'],'section':features.ids[j],'correct':bool(ok)} for r,j,ok in zip(rows,order[:,0],right)]}
    results=[]
    for representation,xx,kxx in [('lexical',oldx,oldkb),('lexical_dense',x,kx)]:
        for kind in ['lr0.01','lr0.1','lr1','trees5']:
            with threadpool_limits(limits=1):
                scores=np.zeros(y.shape)
                for a,b in folds:
                    m=estimator(kind).fit(xx[a].reshape(-1,xx.shape[-1]),y[a].ravel())
                    scores[b]=m.predict_proba(xx[b].reshape(-1,xx.shape[-1]))[:,1].reshape(len(b),-1)
                m=estimator(kind).fit(xx.reshape(-1,xx.shape[-1]),y.ravel())
                external=m.predict_proba(kxx.reshape(-1,kxx.shape[-1]))[:,1].reshape(len(kb),-1)
            r={'representation':representation,'model':kind,'features':xx.shape[-1],
                'public_grouped':report(public,scores,baseline),'kb55':report(kb,external,kbaseline)}
            results.append(r)
            print(json.dumps({**{k:v for k,v in r.items() if k not in ['public_grouped','kb55']},'public_top1':r['public_grouped']['top1'],'kb55_top1':r['kb55']['top1']}),flush=True)
    summary={'scope':'public_grouped_and_prior_kb_development_retrieval_not_tutor_score','api_calls':0,
        'embedding_model':model,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'kb55_used_for_training':False,'results':results}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
