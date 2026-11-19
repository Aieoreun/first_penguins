"""Grouped linear-margin and complement-NB character baselines; API-free."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import public_data


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_linear_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    groups,folds=group_folds(texts,gold)
    models=[('svm01',.1),('svm1',1.),('svm10',10.),('nb01',.1),('nb1',1.)]
    manifest={'scope':'public_grouped_linear_screen_not_private_not_holdout','api_calls':0,
        'ids':[r['id'] for r in rows],'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'models':models,'svm_confidence':'softmax of raw margins, not calibrated probabilities'}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');reports=[]
    for kind,value in models:
        probabilities=np.zeros((len(rows),len(classes)))
        for fold,(a,b) in enumerate(folds):
            with threadpool_limits(limits=1):
                tf=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
                train=tf.fit_transform([texts[i] for i in a]);test=tf.transform([texts[i] for i in b])
                if kind.startswith('svm'):
                    model=LinearSVC(C=value,class_weight='balanced',dual='auto',max_iter=10000,random_state=20261118)
                    model.fit(train,gold[a]);p=softmax(model.decision_function(test),axis=1)
                else:
                    model=ComplementNB(alpha=value);model.fit(train,gold[a]);p=model.predict_proba(test)
            for j,c in enumerate(model.classes_):probabilities[b,np.flatnonzero(classes==c)[0]]=p[:,j]
            print(json.dumps({'model':kind,'fold':fold}),flush=True)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1):raise ValueError('Invalid scores')
        pred=classes[probabilities.argmax(1)];r={'model':kind,'metrics':metrics(gold,pred),
            'matched_coverage':selective(gold,pred,probabilities.max(1))}
        reports.append(r);np.savez_compressed(dest/(kind+'.npz'),classes=classes,probabilities=probabilities)
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(r),flush=True)
    (dest/'COMPLETE').write_text('Grouped linear screen complete.\n')


if __name__=='__main__':main()
