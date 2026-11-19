"""Offline user-history feature ablation on the existing grouped public folds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds, metrics, selective
from pipelines.kimjaehak.classifier_models import public_data


def history(row):
    users=[m['content'] for m in row['messages'] if m['role']=='user']
    return '\n'.join(users[-4:-1])[-1500:]


def vectorizer():
    return TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,
        max_features=60000,sublinear_tf=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True)
    args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_context_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows])
    histories=[history(r) for r in rows];classes=np.unique(gold)
    groups,folds=group_folds(texts,gold)
    configs=[{'name':'history025','weight':.25,'width':32,'alpha':.1},
        {'name':'history050','weight':.5,'width':32,'alpha':.1},
        {'name':'regularized','weight':0,'width':32,'alpha':1.0},
        {'name':'small','weight':0,'width':16,'alpha':.1}]
    manifest={'scope':'public_grouped_oof_not_independent_holdout','api_calls':0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ids':[r['id'] for r in rows],'history':'previous_three_user_turns_last1500chars_no_assistant',
        'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds],'configs':configs}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reports=[]
    for config in configs:
        probs=np.zeros((len(rows),len(classes)));fits=[]
        for fold,(a,b) in enumerate(folds):
            started=time.monotonic()
            with threadpool_limits(limits=1):
                latest=vectorizer();x=latest.fit_transform([texts[i] for i in a]);y=latest.transform([texts[i] for i in b])
                if config['weight']:
                    prior=vectorizer();hx=prior.fit_transform([histories[i] for i in a]);hy=prior.transform([histories[i] for i in b])
                    x=hstack([x,hx*config['weight']],format='csr');y=hstack([y,hy*config['weight']],format='csr')
                model=MLPClassifier(hidden_layer_sizes=(config['width'],),solver='lbfgs',alpha=config['alpha'],
                    max_iter=600,max_fun=30000,random_state=20261118,tol=1e-5)
                model.fit(x,gold[a],sample_weight=compute_sample_weight('balanced',gold[a]))
                p=model.predict_proba(y)
            for j,c in enumerate(model.classes_):probs[b,np.flatnonzero(classes==c)[0]]=p[:,j]
            fits.append({'fold':fold,'seconds':time.monotonic()-started,'features':x.shape[1]})
            print(json.dumps({'config':config['name'],**fits[-1]}),flush=True)
        pred=classes[probs.argmax(axis=1)];conf=probs.max(axis=1)
        report={'config':config,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,conf),'fits':fits}
        reports.append(report)
        np.savez_compressed(dest/(config['name']+'.npz'),probabilities=probs,classes=classes)
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='fits'}),flush=True)
    (dest/'COMPLETE').write_text('Offline public grouped context screen complete.\n')


if __name__=='__main__':main()
