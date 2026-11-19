"""Grouped character-model spacing robustness screen; no API or new labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import public_data


def compact(text):return re.sub(r'\s+','',unicodedata.normalize('NFKC',text).lower())


def vectorizer(compacted=False):
    return TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,
        sublinear_tf=True,preprocessor=compact if compacted else None)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_spacing_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    _,folds=group_folds(texts,gold)
    manifest={'scope':'public_grouped_spacing_cv_not_private_not_holdout','api_calls':0,'models':['compact','dual'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ids':[r['id'] for r in rows],'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'source_at_run.py').write_bytes(Path(__file__).read_bytes());reports=[]
    for kind in manifest['models']:
        probs=np.zeros((len(rows),len(classes)));fits=[]
        for fold,(a,b) in enumerate(folds):
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                tf=vectorizer(True);train=tf.fit_transform([texts[i] for i in a]);test=tf.transform([texts[i] for i in b])
                if kind=='dual':
                    raw=vectorizer();train=hstack([raw.fit_transform([texts[i] for i in a]),train],format='csr')/np.sqrt(2)
                    test=hstack([raw.transform([texts[i] for i in b]),test],format='csr')/np.sqrt(2)
                model=MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',alpha=.1,max_iter=600,max_fun=30000,random_state=20261118,tol=1e-5)
                model.fit(train,gold[a],sample_weight=compute_sample_weight('balanced',gold[a]))
                p=model.predict_proba(test)
            for i,c in enumerate(model.classes_):probs[b,np.flatnonzero(classes==c)[0]]=p[:,i]
            fits.append({'fold':fold,'iterations':int(model.n_iter_),'warnings':sum(issubclass(w.category,ConvergenceWarning) for w in caught)})
            print(json.dumps({'model':kind,'fold':fold}),flush=True)
        pred=classes[probs.argmax(1)]
        r={'model':kind,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,probs.max(1)),'fits':fits};reports.append(r)
        np.savez_compressed(dest/(kind+'.npz'),classes=classes,probabilities=probs)
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(r),flush=True)
    (dest/'COMPLETE').write_text('Public grouped spacing screen complete; no runtime adoption.\n')


if __name__=='__main__':main()
