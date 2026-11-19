"""Separate binary decision head with grouped OOF; no API or new training rows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.jh_classifier_matrix import BLOCKED,group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data


from pipelines.kimjaehak.round7_binary_model import reweight

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r7_binary_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);binary=np.array(['block' if c in BLOCKED else 'allow' for c in gold])
    _,folds=group_folds(texts,gold);ref=ROOT/'experiments/.cache/jh_r7_hybrid_hybrid1'
    prior=json.loads((ref/'manifest.json').read_text())
    if prior['ids']!=[r['id'] for r in rows] or prior['folds']!=[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]:raise ValueError('Reference fold mismatch')
    reference=np.load(ref/'char_mlp.npz');classes=reference['classes'];cp=reference['probabilities']
    manifest={'scope':'public_grouped_binary_head_not_private_not_holdout','api_calls':0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256((ref/'char_mlp.npz').read_bytes()).hexdigest(),
        'ids':prior['ids'],'folds':prior['folds'],'models':['char_lr_c10','char_mlp'],'weights':[.5,1.]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results=[]
    for kind in manifest['models']:
        bp=np.empty(len(rows));fits=[]
        for fold,(a,b) in enumerate(folds):
            model=CategoryModel('char_lr' if kind=='char_lr_c10' else 'char_mlp',c=10).fit([texts[i] for i in a],binary[a])
            bp[b]=model.probabilities([texts[i] for i in b])[:,np.flatnonzero(model.classes=='block')[0]]
            fits.append(model.diagnostics);print(json.dumps({'model':kind,'fold':fold}),flush=True)
        np.savez_compressed(dest/(kind+'_binary.npz'),block_probability=bp)
        for weight in manifest['weights']:
            p=reweight(cp,classes,bp,weight);pred=classes[p.argmax(1)]
            report={'model':kind,'weight':weight,'binary_accuracy':float(np.mean((bp>=.5)==(binary=='block'))),
                'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,p.max(1)),'fits':fits}
            results.append(report);np.savez_compressed(dest/(kind+'_w'+str(weight)+'.npz'),classes=classes,probabilities=p)
            (dest/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(report),flush=True)
    (dest/'COMPLETE').write_text('Grouped binary head screen complete; no runtime promotion.\n')


if __name__=='__main__':main()
