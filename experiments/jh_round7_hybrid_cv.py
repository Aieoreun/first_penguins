"""Grouped OOF fusion of scratch character and organizer-embedding classifiers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds, metrics, selective
from pipelines.kimjaehak.classifier_models import CategoryModel, public_data


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True)
    args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_hybrid_'+args.name);dest.mkdir()
    rows,texts,vectors,embedding_model=public_data()
    gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    groups,folds=group_folds(texts,gold)
    manifest={'scope':'public_grouped_oof_not_independent_holdout','api_calls':0,
        'embedding_model':embedding_model,'seed':20261118,'width':32,'alpha':.1,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ids':[r['id'] for r in rows],'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds],
        'models':['char_mlp','dense_mlp','concat050','concat100'],'blends':[.25,.5,.75]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    stored={};reports=[]
    def record(name,probs):
        assert np.allclose(probs.sum(axis=1),1)
        pred=classes[probs.argmax(axis=1)];conf=probs.max(axis=1)
        report={'name':name,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,conf)}
        stored[name]=probs;reports.append(report)
        np.savez_compressed(dest/(name+'.npz'),probabilities=probs,classes=classes)
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    for kind in manifest['models']:
        probs=np.zeros((len(rows),len(classes)))
        for fold,(a,b) in enumerate(folds):
            if kind in ('char_mlp','dense_mlp'):
                model=CategoryModel(kind).fit([texts[i] for i in a],gold[a],vectors[a])
                p=model.probabilities([texts[i] for i in b],vectors[b]);model_classes=model.classes
            else:
                weight=.5 if kind=='concat050' else 1.0
                with threadpool_limits(limits=1):
                    tf=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
                    x=hstack([tf.fit_transform([texts[i] for i in a]),csr_matrix(CategoryModel._vectors(vectors[a],len(a))*weight)],format='csr')
                    y=hstack([tf.transform([texts[i] for i in b]),csr_matrix(CategoryModel._vectors(vectors[b],len(b))*weight)],format='csr')
                    model=MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',alpha=.1,
                        max_iter=600,max_fun=30000,random_state=20261118,tol=1e-5)
                    model.fit(x,gold[a],sample_weight=compute_sample_weight('balanced',gold[a]))
                    p=model.predict_proba(y);model_classes=model.classes_
            for j,c in enumerate(model_classes):probs[b,np.flatnonzero(classes==c)[0]]=p[:,j]
            print(json.dumps({'name':kind,'fold':fold}),flush=True)
        record(kind,probs)
    for w in manifest['blends']:
        record('blend_dense'+str(w),stored['char_mlp']*(1-w)+stored['dense_mlp']*w)
    cp=classes[stored['char_mlp'].argmax(axis=1)];dp=classes[stored['dense_mlp'].argmax(axis=1)]
    mask=(cp==dp)&(cp!='harmful_content')&(stored['char_mlp'].max(axis=1)>=.8556677733093049)
    (dest/'agreement.json').write_text(json.dumps({'count':int(mask.sum()),
        'errors':int((cp[mask]!=gold[mask]).sum()),'char_threshold':.8556677733093049},indent=2)+'\n')
    (dest/'COMPLETE').write_text('Offline public grouped hybrid screen complete.\n')


if __name__=='__main__':main()
