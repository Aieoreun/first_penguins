"""Public grouped OOF: vary class balancing separately in the two input heads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_binary_model import BLOCKED,reweight
from experiments.jh_classifier_matrix import group_folds,metrics,selective


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_weight_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold)
    decisions=np.array(['block' if c in BLOCKED else 'allow' for c in gold])
    _,folds=group_folds(texts,gold);cache=ROOT/'experiments/.cache'
    source_cat=cache/'jh_r7_hybrid_hybrid1/char_mlp.npz';source_bin=cache/'jh_r7_binary_screen1/char_mlp_binary.npz'
    sources={}
    for path in [source_cat,source_bin]:
        m=json.loads((path.parent/'manifest.json').read_text())
        assert m['ids']==[r['id'] for r in rows]
        assert m['folds']==[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    z=np.load(source_cat);assert np.array_equal(z['classes'],classes)
    category={1.:z['probabilities']};binary={1.:np.load(source_bin)['block_probability']};fits=[]
    for power in [0.,.5]:
        category[power]=np.zeros((len(rows),len(classes)));binary[power]=np.zeros(len(rows))
        for fold,(a,b) in enumerate(folds):
            for kind,labels in [('category',gold),('binary',decisions)]:
                weights=compute_sample_weight('balanced',labels[a])**power
                weights=weights/weights.mean()
                # This offline patch changes only fit weights; every vectorizer,
                # optimizer, seed and architecture setting remains CategoryModel.
                with patch('pipelines.kimjaehak.classifier_models.compute_sample_weight',return_value=weights):
                    model=CategoryModel('char_mlp').fit([texts[i] for i in a],labels[a])
                p=model.probabilities([texts[i] for i in b])
                if kind=='category':
                    assert np.array_equal(model.classes,classes)
                    category[power][b]=p
                else:binary[power][b]=p[:,int(np.flatnonzero(model.classes=='block')[0])]
                fits.append({'power':power,'fold':fold,'kind':kind,**model.diagnostics})
            print('fit',power,fold,flush=True)
    reports=[]
    for cp in [1.,0.,.5]:
        for bp in [1.,0.,.5]:
            p=reweight(category[cp],classes,binary[bp],1.);pred=classes[p.argmax(1)]
            r={'category_power':cp,'binary_power':bp,'metrics':metrics(gold,pred),
               'matched_coverage':selective(gold,pred,p.max(1))};reports.append(r)
            np.savez_compressed(dest/(f'category{cp}_binary{bp}.npz'),classes=classes,probabilities=p)
            print(cp,bp,r['matched_coverage']['0.5'],flush=True)
    out={'scope':'posthoc_public_grouped_OOF_development_not_holdout','api_calls':0,
         'ids':[r['id'] for r in rows],'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds],
         'sources':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'fits':fits,'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    (dest/'COMPLETE').write_text('Weight screening complete.\n')


if __name__=='__main__':main()
