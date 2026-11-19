"""Group-held-out screen of organizer-generated training paraphrases.

Every original and its augmented text remain in the same fold. Existing
development paraphrases are used only in this offline feasibility study, not
silently promoted into deployment training or claimed as independent holdout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import public_data
from experiments.jh_transfer_probe import build_items,normalized


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_augmentation_'+args.name);dest.mkdir()
    source=ROOT/'experiments/.cache/jh_augment_public2_1/paraphrases.jsonl'
    primary,provenance=build_items(source)
    alternative,_=build_items(source,True)
    rows,texts,_,_=public_data();indices={r['id']:i for i,r in enumerate(rows)}
    gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    groups,folds=group_folds(texts,gold)
    original_norms={normalized(t) for t in texts}
    aug={indices[r['source_id']]:r['messages'][-1]['content'] for r in primary
         if normalized(r['messages'][-1]['content']) not in original_norms}
    alt_gold=np.array([r['labels']['category'] for r in alternative])
    alt_parent=np.array([indices[r['source_id']] for r in alternative])
    manifest={'scope':'public_parent_grouped_augmented_training_feasibility_not_private',
        'api_calls':0,'deployment_training_changed':False,'labels_inherited_not_independently_reviewed':True,
        'generation_model':provenance['model'],'paraphrases_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'augmentation_count':len(aug),'original_count':len(rows),'alternative_count':len(alternative),
        'weights':[0,.25,.5,1.],'seed':20261118,'width':32,'alpha':.1,
        'ids':[r['id'] for r in rows],'alternative_ids':[r['id'] for r in alternative],
        'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reports=[]
    for weight in manifest['weights']:
        probabilities=np.zeros((len(rows),len(classes)))
        alt_probabilities=np.zeros((len(alternative),len(classes)))
        for fold,(a,b) in enumerate(folds):
            assert not set(groups[a])&set(groups[b])
            train=[texts[i] for i in a];labels=list(gold[a])
            parent_weights=compute_sample_weight('balanced',gold[a])
            weights=list(parent_weights)
            if weight:
                for i,w in zip(a,parent_weights):
                    if i in aug:
                        train.append(aug[i]);labels.append(gold[i]);weights.append(w*weight)
            weights=np.array(weights);weights*=len(a)/weights.sum()
            held=np.flatnonzero(np.isin(alt_parent,b))
            with threadpool_limits(limits=1):
                tf=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,
                    max_features=60000,sublinear_tf=True)
                x=tf.fit_transform(train)
                model=MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',alpha=.1,
                    max_iter=600,max_fun=30000,random_state=20261118,tol=1e-5)
                model.fit(x,labels,sample_weight=weights)
                p=model.predict_proba(tf.transform([texts[i] for i in b]))
                q=model.predict_proba(tf.transform([alternative[i]['messages'][-1]['content'] for i in held]))
                for j,c in enumerate(model.classes_):
                    col=int(np.flatnonzero(classes==c)[0]);probabilities[b,col]=p[:,j];alt_probabilities[held,col]=q[:,j]
            print(json.dumps({'weight':weight,'fold':fold,'train':len(train)}),flush=True)
        result={'augmentation_weight':weight}
        for name,p,y in [('original',probabilities,gold),('alternative',alt_probabilities,alt_gold)]:
            assert np.allclose(p.sum(axis=1),1)
            pred=classes[p.argmax(1)];conf=p.max(1)
            result[name]={'metrics':metrics(y,pred),'matched_coverage':selective(y,pred,conf)}
        reports.append(result)
        np.savez_compressed(dest/f'weight{weight}.npz',classes=classes,probabilities=probabilities,alternative=alt_probabilities)
        (dest/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    (dest/'COMPLETE').write_text('Offline grouped augmentation screen complete.\n')


if __name__=='__main__':main()
