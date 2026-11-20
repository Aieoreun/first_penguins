"""Common-fallback input replay for binary/category combinations; no API calls."""
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pipelines import load
load('jh_r7_binary_guard')
from core.schema import POLICY
from experiments.jh_classifier_matrix import metrics
from experiments.jh_round7_gate_replay import gate_label
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_guard import BoundaryModel
from pipelines.kimjaehak.round7_binary_model import BinaryAdjustedModel,BLOCKED
from pipelines.kimjaehak.learned_harmful import train_harmful_classifier

def main():
    cache=ROOT/'experiments/.cache';dest=cache/'jh_r8_cross_replay1';dest.mkdir()
    source=cache/'jh_guard_fusion_r7_g07';prior=cache/'jh_r7_replay_primary1'
    raw=(source/'inputs.jsonl').read_bytes();items=[json.loads(s) for s in raw.splitlines()]
    plan=json.loads((source/'plan.json').read_text());vm=json.loads((prior/'manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==plan['dataset_sha256']==vm['dataset_sha256']
    fallbackfile=source/'r7_llm_input/responses.jsonl';fallback={r['id']:r['response']['input'] for r in map(json.loads,fallbackfile.read_text().splitlines())}
    assert set(fallback)=={r['id'] for r in items}
    texts=[next(m['content'] for m in reversed(i['messages']) if m['role']=='user') for i in items]
    train,tt,vectors,embedding=public_data();assert embedding==vm['embedding_model'];tv=np.load(prior/'vectors.npz')['vectors'];CategoryModel._vectors(tv,len(items))
    labels=np.array([r['labels']['category'] for r in train]);gold=np.array([r['labels']['category'] for r in items])
    binary=CategoryModel('char_mlp').fit(tt,['block' if c in BLOCKED else 'allow' for c in labels])
    hv,hm=train_harmful_classifier(1000);hp=hm.predict_proba(hv.transform(texts));sub=[(str(hm.classes_[p.argmax()]),float(p.max())) for p in hp]
    models={'baseline':CategoryModel('char_mlp'),'boundary':BoundaryModel('char_mlp'),'dense':CategoryModel('dense_mlp')}
    thresholds={'baseline':.8776460484899311,'boundary':.889530520789236,'dense':.8820789131634711};reports=[];baseline=None
    for name,category in models.items():
        category.fit(tt,labels,vectors);m=BinaryAdjustedModel(category,binary);p=m.probabilities(texts,tv);pred=[];direct=[]
        for i,item in enumerate(items):
            label=gate_label(texts[i],m.classes,p[i],thresholds[name],sub[i]);direct.append(label is not None)
            pred.append(fallback[item['id']]['category'] if label is None else POLICY[label].category.value)
        pred=np.array(pred);direct=np.array(direct)
        if baseline is None:baseline=pred
        report={'model':name,'threshold':thresholds[name],'metrics':metrics(gold,pred),'direct':int(direct.sum()),'direct_errors':int((pred[direct]!=gold[direct]).sum()),'corrected_vs_B':int(((pred==gold)&(baseline!=gold)).sum()),'regressed_vs_B':int(((pred!=gold)&(baseline==gold)).sum())};reports.append(report)
        (dest/(name+'.json')).write_text(json.dumps([{'id':r['id'],'category':str(pred[i]),'direct':bool(direct[i])} for i,r in enumerate(items)],indent=2)+'\n');print(json.dumps(report),flush=True)
    out={'scope':'public_frozen_fallback_input_only_not_live_score_not_holdout','api_calls':0,'dataset_sha256':plan['dataset_sha256'],'fallback_sha256':hashlib.sha256(fallbackfile.read_bytes()).hexdigest(),'embedding_model':embedding,'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Replay complete.\n')
if __name__=='__main__':main()
