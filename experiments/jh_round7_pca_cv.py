"""Scratch PCA of provided embeddings, grouped OOF; no API or pretrained weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import public_data


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True)
    ap.add_argument('--max-iter',type=int,default=600)
    args=ap.parse_args()
    if not 100<=args.max_iter<=5000:ap.error('Bounded fit iteration limit required')
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_pca_'+args.name);dest.mkdir()
    rows,texts,vectors,embedding_model=public_data();gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    groups,folds=group_folds(texts,gold);vectors=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    reference=ROOT/'experiments/.cache/jh_r7_hybrid_hybrid1'
    prior=json.loads((reference/'manifest.json').read_text())
    current_folds=[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]
    if prior['ids']!=[r['id'] for r in rows] or prior['folds']!=current_folds:
        raise ValueError('Character OOF reference groups differ')
    char=np.load(reference/'char_mlp.npz')
    if not np.array_equal(char['classes'],classes):raise ValueError('Class order differs')
    cp=char['probabilities']
    manifest={'scope':'public_grouped_pca_scratch_cv_not_private_not_holdout','api_calls':0,
        'embedding_model':embedding_model,'seed':20261118,'max_iter':args.max_iter,'models':[[32,False],[64,False],[32,True],[64,True]],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ids':prior['ids'],'folds':current_folds,
        'character_reference_sha256':hashlib.sha256((reference/'char_mlp.npz').read_bytes()).hexdigest()}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');reports=[]
    for width,whiten in manifest['models']:
        kind=f'pca{width}'+('_white' if whiten else '')
        probabilities=np.zeros((len(rows),len(classes)));fits=[]
        for fold,(a,b) in enumerate(folds):
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                pca=PCA(n_components=width,whiten=whiten,svd_solver='full')
                train=pca.fit_transform(vectors[a]);test=pca.transform(vectors[b])
                model=MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',alpha=.1,
                    max_iter=args.max_iter,max_fun=max(30000,args.max_iter*50),random_state=20261118,tol=1e-5)
                model.fit(train,gold[a],sample_weight=compute_sample_weight('balanced',gold[a]))
                p=model.predict_proba(test)
            fits.append({'fold':fold,'iterations':int(model.n_iter_),'loss':float(model.loss_),
                'training_accuracy':float(model.score(train,gold[a])),
                'convergence_warnings':sum(issubclass(w.category,ConvergenceWarning) for w in caught)})
            for j,c in enumerate(model.classes_):probabilities[b,np.flatnonzero(classes==c)[0]]=p[:,j]
            print(json.dumps({'model':kind,'fold':fold}),flush=True)
        for name,probs in [(kind,probabilities),(kind+'_char75',.75*cp+.25*probabilities)]:
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1):raise ValueError('Invalid probabilities')
            pred=classes[probs.argmax(1)];confidence=probs.max(1)
            report={'model':name,'fits':fits,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,confidence)}
            reports.append(report);np.savez_compressed(dest/(name+'.npz'),classes=classes,probabilities=probs)
            (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(report),flush=True)
    (dest/'COMPLETE').write_text('Grouped PCA screening complete; no runtime adoption.\n')


if __name__=='__main__':main()
