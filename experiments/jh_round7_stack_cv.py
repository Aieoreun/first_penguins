"""Nested grouped evaluation of a small learned character/dense combiner.

Outer test labels never fit the base models, vocabulary, or combiner. All data
are the bundled public examples; cached organizer embeddings are fixed features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds, metrics, selective
from pipelines.kimjaehak.classifier_models import CategoryModel, public_data


def pair_predictions(texts,labels,vectors,train,test,classes):
    parts=[]
    for kind in ['char_mlp','dense_mlp']:
        m=CategoryModel(kind).fit([texts[i] for i in train],labels[train],vectors[train])
        p=m.probabilities([texts[i] for i in test],vectors[test])
        aligned=np.zeros((len(test),len(classes)))
        for j,c in enumerate(m.classes):aligned[:,np.flatnonzero(classes==c)[0]]=p[:,j]
        parts.append(aligned)
    return np.hstack(parts)


def features(p,kind):
    return np.log(np.clip(p,1e-6,1)) if kind=='log' else p


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True)
    ap.add_argument('--class-weight',choices=['balanced','none'],default='balanced')
    ap.add_argument('--reuse-base')
    args=ap.parse_args()
    if any(n and not n.replace('_','').isalnum() for n in (args.name,args.reuse_base)):ap.error('Simple cache names required')
    dest=ROOT/'experiments/.cache'/('jh_r7_stack_'+args.name);dest.mkdir()
    rows,texts,vectors,embedding_model=public_data();gold=np.array([r['labels']['category'] for r in rows])
    classes=np.unique(gold);groups,folds=group_folds(texts,gold)
    configs=[{'features':f,'c':c,'class_weight':None if args.class_weight=='none' else 'balanced'}
        for f in ['raw','log'] for c in [.1,1.0]]
    manifest={'scope':'nested_public_grouped_oof_not_independent_holdout','api_calls':0,
        'embedding_model':embedding_model,'outer_folds':5,'inner_folds':3,'configs':configs,
        'ids':[r['id'] for r in rows],'classes':classes.tolist(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    previous=ROOT/'experiments/.cache'/('jh_r7_stack_'+args.reuse_base) if args.reuse_base else None
    if previous:
        old=json.loads((previous/'manifest.json').read_text())
        if any(old[k]!=manifest[k] for k in ('embedding_model','ids','classes','folds')):
            raise ValueError('Incompatible base fold cache')
        manifest['reused_base_source_sha256']=old['source_sha256']
        manifest['reused_base']=args.reuse_base
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=[np.zeros((len(rows),len(classes))) for c in configs]
    outer_predictions=np.zeros((len(rows),len(classes)*2))
    for fold,(a,b) in enumerate(folds):
        if previous:
            cache=np.load(previous/f'fold{fold}.npz')
            assert np.array_equal(a,cache['train']) and np.array_equal(b,cache['test'])
            inside,outside=cache['inner'],cache['outer']
        else:
            inner=StratifiedGroupKFold(3,shuffle=True,random_state=20261120)
            inside=np.zeros((len(a),len(classes)*2));seen=np.zeros(len(a),int)
            for k,(ia,ib) in enumerate(inner.split(a,gold[a],groups[a])):
                assert not (set(groups[a[ia]])&set(groups[a[ib]]))
                inside[ib]=pair_predictions(texts,gold,vectors,a[ia],a[ib],classes);seen[ib]+=1
                print(json.dumps({'outer':fold,'inner':k,'done':True}),flush=True)
            assert np.all(seen==1)
            outside=pair_predictions(texts,gold,vectors,a,b,classes)
        outer_predictions[b]=outside
        np.savez_compressed(dest/f'fold{fold}.npz',inner=inside,outer=outside,train=a,test=b)
        for j,config in enumerate(configs):
            with threadpool_limits(limits=1):
                m=LogisticRegression(C=config['c'],max_iter=1000,class_weight=config['class_weight'])
                m.fit(features(inside,config['features']),gold[a])
                p=m.predict_proba(features(outside,config['features']))
            for cidx,c in enumerate(m.classes_):result[j][b,np.flatnonzero(classes==c)[0]]=p[:,cidx]
        print(json.dumps({'outer':fold,'complete':True}),flush=True)
    reports=[]
    for config,p in zip(configs,result):
        pred=classes[p.argmax(1)];conf=p.max(1)
        report={'config':config,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,conf)}
        reports.append(report)
        np.savez_compressed(dest/(config['features']+'_'+str(config['c'])+'.npz'),probabilities=p,classes=classes)
        print(json.dumps(report),flush=True)
    np.savez_compressed(dest/'base_outer_predictions.npz',probabilities=outer_predictions,classes=classes)
    (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
    (dest/'COMPLETE').write_text('Nested grouped public stack screen complete.\n')


if __name__=='__main__':main()
