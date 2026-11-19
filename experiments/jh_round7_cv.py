"""Offline grouped public-data screen for boundary-aware word/character features."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds, metrics, selective
from pipelines.kimjaehak.classifier_models import public_data


def features(kind):
    char = TfidfVectorizer(analyzer='char_wb' if kind=='boundary_mlp' else 'char',
        ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
    if kind=='boundary_mlp': return char
    return FeatureUnion([('char',char),('word',TfidfVectorizer(ngram_range=(1,2),
        token_pattern=r'(?u)\b\w+\b',min_df=2,max_features=20000,sublinear_tf=True))])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple experiment name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_cv_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data()
    gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    groups,folds=group_folds(texts,gold)
    manifest={'scope':'public_grouped_oof_feature_screen_not_private_score','api_calls':0,
        'training':'bundled_public_only','independent_holdout':False,'group_count':len(set(groups)),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ids':[r['id'] for r in rows], 'classes':classes.tolist(),
        'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds],
        'seed':20261118,'mlp_width':32,'mlp_alpha':.1,'lr_c':100,
        'models':['boundary_mlp','wordchar_lr','wordchar_mlp']}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reports=[]
    for kind in manifest['models']:
        probs=np.zeros((len(rows),len(classes)));fits=[]
        for fold,(a,b) in enumerate(folds):
            started=time.monotonic();vectorizer=features(kind)
            with threadpool_limits(limits=1):
                train=vectorizer.fit_transform([texts[i] for i in a])
                if kind.endswith('_lr'):
                    model=LogisticRegression(C=100,max_iter=1000,class_weight='balanced')
                    model.fit(train,gold[a])
                else:
                    model=MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',alpha=.1,
                        max_iter=600,max_fun=30000,random_state=20261118,tol=1e-5)
                    model.fit(train,gold[a],sample_weight=compute_sample_weight('balanced',gold[a]))
                pred=model.predict_proba(vectorizer.transform([texts[i] for i in b]))
            for j,c in enumerate(model.classes_):probs[b,np.flatnonzero(classes==c)[0]]=pred[:,j]
            fits.append({'fold':fold,'seconds':time.monotonic()-started,'features':train.shape[1]})
            print(json.dumps({'kind':kind,**fits[-1]}),flush=True)
        assert np.allclose(probs.sum(axis=1),1)
        pred=classes[probs.argmax(axis=1)];conf=probs.max(axis=1)
        report={'kind':kind,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,conf),'fits':fits}
        reports.append(report)
        np.savez_compressed(dest/(kind+'.npz'),probabilities=probs,classes=classes)
        (dest/(kind+'_predictions.json')).write_text(json.dumps([
            {'id':row['id'],'gold':str(gold[i]),'prediction':str(pred[i]),'confidence':float(conf[i])}
            for i,row in enumerate(rows)],indent=2)+'\n')
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='fits'}),flush=True)
    (dest/'COMPLETE').write_text('Offline grouped public-data feature screen complete.\n')


if __name__=='__main__':main()
