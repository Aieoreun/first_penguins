"""Grouped public output-safety triage screen; candidate text only, no API."""
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import f1_score
from experiments.jh_classifier_matrix import group_folds
from pipelines.kimjaehak.classifier_models import CategoryModel

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_safety_'+args.name);dest.mkdir()
    source=ROOT/'public/guard.jsonl';rows=[r for r in map(json.loads,source.read_text().splitlines()) if isinstance(r.get('candidate_answer'),str) and r['candidate_answer'].strip()]
    texts=[r['candidate_answer'] for r in rows];gold=np.array(['safe' if r['labels']['safe'] else 'unsafe' for r in rows]);groups,folds=group_folds(texts,gold)
    manifest={'scope':'grouped_public_candidate_safety_routing_not_holdout','public_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'api_calls':0,'ids':[r['id'] for r in rows],'groups':groups.tolist(),'folds':[{'train':a.tolist(),'test':b.tolist()} for a,b in folds]}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results=[]
    for kind,c in [('char_lr',1),('char_lr',10),('char_lr',100),('char_mlp',3000)]:
        p=np.empty(len(rows));fits=[]
        for a,b in folds:
            m=CategoryModel(kind,c=c).fit([texts[i] for i in a],gold[a]);p[b]=m.probabilities([texts[i] for i in b])[:,np.flatnonzero(m.classes=='unsafe')[0]];fits.append(m.diagnostics)
        name=kind+str(c);thresholds=[]
        for threshold in [.1,.2,.3,.4,.5,.7,.9]:
            pred=p>=threshold;truth=gold=='unsafe';thresholds.append({'threshold':threshold,'selected':int(pred.sum()),'false_alarm':int((pred&~truth).sum()),'miss':int((~pred&truth).sum()),'unsafe_f1':float(f1_score(truth,pred))})
        r={'model':name,'fits':fits,'thresholds':thresholds};results.append(r);np.savez_compressed(dest/(name+'.npz'),probabilities=p);print(json.dumps(r),flush=True)
    (dest/'results.json').write_text(json.dumps({'manifest':manifest,'results':results},indent=2)+'\n');(dest/'COMPLETE').write_text('Safety triage screen complete.\n')
if __name__=='__main__':main()
