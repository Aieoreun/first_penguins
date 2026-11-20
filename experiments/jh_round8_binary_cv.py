"""Screen binary-head stability on the same grouped public folds, without API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.jh_classifier_matrix import group_folds,metrics,selective
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_binary_model import BLOCKED,reweight

CONFIGS={'seed19':{'seed':20261119},'seed20':{'seed':20261120},
         'width16':{'width':16},'width64':{'width':64},'alpha1':{'alpha':1.}}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_binary_'+args.name);dest.mkdir()
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);binary=np.array(['block' if x in BLOCKED else 'allow' for x in gold])
    _,folds=group_folds(texts,gold);ref=ROOT/'experiments/.cache/jh_r7_hybrid_hybrid1';old=ROOT/'experiments/.cache/jh_r7_binary_screen1'
    manifest=json.loads((ref/'manifest.json').read_text());oldmeta=json.loads((old/'manifest.json').read_text())
    ids=[r['id'] for r in rows];fs=[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]
    if any(m['ids']!=ids or m['folds']!=fs for m in [manifest,oldmeta]):raise ValueError('Fold identity mismatch')
    cpfile=ref/'char_mlp.npz';bpfile=old/'char_mlp_binary.npz'
    base=np.load(cpfile);classes=base['classes'];cp=base['probabilities'];bps={'baseline':np.load(bpfile)['block_probability']}
    meta={'scope':'public_grouped_binary_ablation_not_holdout','api_calls':0,'configs':CONFIGS,'ids':ids,'folds':fs,
          'reference_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cpfile,bpfile]},
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');results=[]
    def record(name,bp,fits):
        p=reweight(cp,classes,bp,1.);pred=classes[p.argmax(1)]
        r={'model':name,'metrics':metrics(gold,pred),'matched_coverage':selective(gold,pred,p.max(1)),'fits':fits}
        results.append(r);np.savez_compressed(dest/(name+'.npz'),classes=classes,probabilities=p,block_probability=bp)
        (dest/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(r),flush=True)
    record('baseline',bps['baseline'],[])
    for name,config in CONFIGS.items():
        bp=np.empty(len(rows));fits=[]
        for fold,(a,b) in enumerate(folds):
            m=CategoryModel('char_mlp',**config).fit([texts[i] for i in a],binary[a])
            bp[b]=m.probabilities([texts[i] for i in b])[:,np.flatnonzero(m.classes=='block')[0]]
            fits.append(m.diagnostics);print(json.dumps({'model':name,'fold':fold}),flush=True)
        bps[name]=bp;record(name,bp,fits)
    for name,members in {'seed3':['baseline','seed19','seed20'],'width3':['baseline','width16','width64'],'regularized2':['baseline','alpha1']}.items():
        record(name,np.mean([bps[x] for x in members],axis=0),[])
    (dest/'COMPLETE').write_text('Grouped binary screen complete; no runtime promotion.\n')

if __name__=='__main__':main()
