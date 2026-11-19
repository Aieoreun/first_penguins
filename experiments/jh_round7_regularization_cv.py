"""Grouped scratch character-MLP regularization/feature ablations, API-free."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import public_data

CONFIGS={'alpha1':{'alpha':1.},'alpha10':{'alpha':10.},
         'min1':{'min_df':1},'grams36':{'ngram_range':(3,6)}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    ap.add_argument('--models',nargs='+',choices=list(CONFIGS),default=list(CONFIGS))
    args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_regularization_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    groups,folds=group_folds(texts,gold)
    manifest={'scope':'grouped_public_scratch_cv_not_independent_holdout','api_calls':0,
        'seed':20261118,'group_count':len(set(groups)),'models':args.models,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'public_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'public').glob('*.jsonl'))},
        'ids':[r['id'] for r in rows],'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');reports=[]
    for kind in args.models:
        config={'alpha':.1,'min_df':2,'ngram_range':(2,5),**CONFIGS[kind]}
        probabilities=np.zeros((len(rows),len(classes)))
        for fold,(a,b) in enumerate(folds):
            with threadpool_limits(limits=1):
                vectorizer=TfidfVectorizer(analyzer='char',ngram_range=config['ngram_range'],
                    min_df=config['min_df'],max_features=60000,sublinear_tf=True)
                train=vectorizer.fit_transform([texts[i] for i in a])
                model=MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',alpha=config['alpha'],
                    max_iter=600,max_fun=30000,random_state=20261118,tol=1e-5)
                model.fit(train,gold[a],sample_weight=compute_sample_weight('balanced',gold[a]))
                p=model.predict_proba(vectorizer.transform([texts[i] for i in b]))
            for j,c in enumerate(model.classes_):probabilities[b,np.flatnonzero(classes==c)[0]]=p[:,j]
            print(json.dumps({'model':kind,'fold':fold}),flush=True)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1):
            raise ValueError('Invalid probabilities')
        pred=classes[probabilities.argmax(1)];confidence=probabilities.max(1)
        result={'model':kind,'config':config,'metrics':metrics(gold,pred),
                'matched_coverage':selective(gold,pred,confidence)}
        reports.append(result);np.savez_compressed(dest/(kind+'.npz'),classes=classes,probabilities=probabilities)
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(result),flush=True)
    (dest/'COMPLETE').write_text('Offline grouped regularization screen complete.\n')


if __name__=='__main__':main()
