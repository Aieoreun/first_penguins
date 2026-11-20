"""Grouped public OOF: separately learn categories within each decision group."""
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pipelines import load
load('jh_r7_binary_guard')
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_guard import BoundaryModel
from pipelines.kimjaehak.round7_binary_model import BLOCKED


def fit_children(texts,labels,kind):
    models={}
    for blocked in [False,True]:
        idx=[i for i,y in enumerate(labels) if (y in BLOCKED)==blocked]
        factory=BoundaryModel if kind=='boundary' else CategoryModel
        model=factory('char_lr' if kind=='lr' else 'char_mlp',c=100).fit([texts[i] for i in idx],[labels[i] for i in idx])
        models[blocked]=model
    return models


def compose(models,texts,bp,classes):
    p=np.zeros((len(texts),len(classes)))
    for blocked,model in models.items():
        q=model.probabilities(texts);mass=bp if blocked else 1-bp
        for j,c in enumerate(model.classes):p[:,np.flatnonzero(classes==c)[0]]=q[:,j]*mass
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1):raise ValueError('Invalid hierarchical distribution')
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_hierarchy_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);classes=np.unique(gold);_,folds=group_folds(texts,gold)
    ref=ROOT/'experiments/.cache/jh_r7_binary_screen1';prior=json.loads((ref/'manifest.json').read_text())
    assert prior['ids']==[r['id'] for r in rows] and prior['folds']==[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]
    binary=ref/'char_mlp_binary.npz';bp=np.load(binary)['block_probability']
    meta={'scope':'grouped_public_conditional_heads_not_holdout','api_calls':0,'ids':prior['ids'],'folds':prior['folds'],'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(dest/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');results=[]
    for kind in ['mlp','boundary','lr']:
        p=np.zeros((len(rows),len(classes)));fits=[]
        for fold,(a,b) in enumerate(folds):
            models=fit_children([texts[i] for i in a],gold[a],kind)
            p[b]=compose(models,[texts[i] for i in b],bp[b],classes)
            fits.append({str(k):getattr(m,'diagnostics',None) for k,m in models.items()});print(kind,fold,flush=True)
        pred=classes[p.argmax(1)];r={'model':kind,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,p.max(1)),'fits':fits};results.append(r)
        np.savez_compressed(dest/(kind+'.npz'),classes=classes,probabilities=p);(dest/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(r),flush=True)
    (dest/'COMPLETE').write_text('Conditional group training completed.\n')
if __name__=='__main__':main()
