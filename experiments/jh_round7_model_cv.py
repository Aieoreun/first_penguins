"""Fixed-group scratch classifier alternatives; no API or pretrained models."""
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
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import public_data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    ap.add_argument('--models',nargs='+',choices=['char_sqrt','char_unweighted','char_wide',
        'char_svm1','char_svm10','char_svm100','dense_svm10'],required=True)
    args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_models_'+args.name);dest.mkdir()
    rows,texts,vectors,embedding_model=public_data()
    gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    groups,folds=group_folds(texts,gold)
    manifest={'scope':'public_grouped_scratch_model_screen_not_private_score','api_calls':0,
        'seed':20261118,'embedding_model':embedding_model,'models':args.models,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ids':[r['id'] for r in rows],'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reports=[]
    for kind in args.models:
        probabilities=np.zeros((len(rows),len(classes)))
        for fold,(a,b) in enumerate(folds):
            with threadpool_limits(limits=1):
                if kind.startswith('char'):
                    tf=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,
                        max_features=60000,sublinear_tf=True)
                    x=tf.fit_transform([texts[i] for i in a]);test=tf.transform([texts[i] for i in b])
                else:
                    x=vectors[a]/np.linalg.norm(vectors[a],axis=1,keepdims=True)
                    test=vectors[b]/np.linalg.norm(vectors[b],axis=1,keepdims=True)
                if 'svm' in kind:
                    # Platt calibration sees only this outer fold's training data.
                    model=SVC(C=float(kind.split('svm')[-1]),kernel='rbf',gamma=1.,
                              class_weight='balanced',probability=True,random_state=20261118)
                    model.fit(x,gold[a])
                else:
                    power={'char_sqrt':.5,'char_unweighted':0.,'char_wide':1.}[kind]
                    weights=compute_sample_weight('balanced',gold[a])**power
                    weights*=len(a)/weights.sum()
                    model=MLPClassifier(hidden_layer_sizes=(64 if kind=='char_wide' else 32,),
                        solver='lbfgs',alpha=.1,max_iter=600,max_fun=30000,
                        random_state=20261118,tol=1e-5)
                    model.fit(x,gold[a],sample_weight=weights)
                p=model.predict_proba(test)
                for j,c in enumerate(model.classes_):probabilities[b,np.flatnonzero(classes==c)[0]]=p[:,j]
            print(json.dumps({'model':kind,'fold':fold}),flush=True)
        assert np.allclose(probabilities.sum(axis=1),1)
        pred=classes[probabilities.argmax(1)];confidence=probabilities.max(1)
        result={'model':kind,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,confidence)}
        reports.append(result)
        np.savez_compressed(dest/(kind+'.npz'),classes=classes,probabilities=probabilities)
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    (dest/'COMPLETE').write_text('Offline scratch alternatives completed.\n')


if __name__=='__main__':main()
