"""Matched-coverage group gate screening from frozen public OOF probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pipelines.kimjaehak.classifier_models import public_data
from pipelines.kimjaehak.round7_binary_model import BLOCKED
from experiments.jh_classifier_matrix import metrics

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_group_gate_'+args.name);dest.mkdir()
    root=ROOT/'experiments/.cache/jh_r7_binary_screen1';source=root/'char_mlp_w1.0.npz';meta=json.loads((root/'manifest.json').read_text())
    rows,_,_,_=public_data();assert meta['ids']==[r['id'] for r in rows]
    z=np.load(source);p=z['probabilities'];classes=z['classes'];gold=np.array([r['labels']['category'] for r in rows]);pred=classes[p.argmax(1)];confidence=p.max(1)
    blocked=np.array([x in BLOCKED for x in pred]);eligible=pred!='harmful_content'
    bi=np.flatnonzero(eligible&blocked);ai=np.flatnonzero(eligible&~blocked)
    bi=bi[np.argsort(-confidence[bi],kind='stable')];ai=ai[np.argsort(-confidence[ai],kind='stable')]
    results=[]
    for n in [0,10,20,30,40,50,60,70,80]:
        if n>len(bi) or 200-n>len(ai):continue
        ix=np.r_[bi[:n],ai[:200-n]]
        bth=float(confidence[bi[n-1]]) if n else 1.000001;ath=float(confidence[ai[200-n-1]])
        pb=np.array([x in BLOCKED for x in pred[ix]]);gb=np.array([x in BLOCKED for x in gold[ix]])
        results.append({'block_slots':n,'allow_slots':200-n,'block_threshold':bth,'allow_threshold':ath,
                        'category_errors':int((pred[ix]!=gold[ix]).sum()),'decision_errors':int((pb!=gb).sum()),
                        'non_normal_category_errors':int(((pred[ix]!=gold[ix])&(gold[ix]!='normal')).sum()),
                        'metrics':metrics(gold[ix],pred[ix])})
    out={'scope':'same_public_grouped_OOF_threshold_screen_not_holdout','api_calls':0,'count':400,
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'results':results}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Offline screening complete.\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
