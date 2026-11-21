"""API-free margin/entropy gates using fixed OOF probabilities and frozen fallback."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_binary_model import BinaryAdjustedModel,BLOCKED
from pipelines.kimjaehak.learned_harmful import train_harmful_classifier
from experiments.jh_round7_gate_replay import gate_label
from experiments.jh_classifier_matrix import metrics
from core.schema import POLICY


def confidence(p,kind):
    if kind=='maximum':return p.max(1)
    if kind=='margin':return np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]
    if kind=='entropy':return 1+(p*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(p.shape[1])
    raise ValueError(kind)


def main():
    cache=ROOT/'experiments/.cache';dest=cache/'jh_r8_gate_confidence1';dest.mkdir()
    data=np.load(cache/'jh_r8_binary_screen1/baseline.npz');classes=data['classes'];p=data['probabilities']
    rows,texts,_,_=public_data();gold=np.array([r['labels']['category'] for r in rows]);pred=classes[p.argmax(1)]
    manifest=json.loads((cache/'jh_r8_binary_screen1/manifest.json').read_text())
    assert manifest['ids']==[r['id'] for r in rows]
    category=CategoryModel('char_mlp').fit(texts,gold);binary=CategoryModel('char_mlp').fit(texts,['block' if c in BLOCKED else 'allow' for c in gold]);model=BinaryAdjustedModel(category,binary)
    source=cache/'jh_guard_fusion_r7_g07';items=list(map(json.loads,(source/'inputs.jsonl').read_text().splitlines()))
    fallback={r['id']:r['response']['input']['category'] for r in map(json.loads,(source/'r7_llm_input/responses.jsonl').read_text().splitlines())}
    assert set(fallback)=={r['id'] for r in items}
    itexts=[next(m['content'] for m in reversed(r['messages']) if m['role']=='user') for r in items];q=model.probabilities(itexts);igold=np.array([r['labels']['category'] for r in items])
    hv,hm=train_harmful_classifier(1000);sub=[(str(hm.classes_[v.argmax()]),float(v.max())) for v in hm.predict_proba(hv.transform(itexts))]
    reports=[];base=None
    for kind in ['maximum','margin','entropy']:
        conf=confidence(p,kind);eligible=np.flatnonzero(pred!='harmful_content');chosen=eligible[np.argsort(-conf[eligible],kind='stable')][:200];threshold=float(conf[chosen].min())
        qc=confidence(q,kind);output=[];direct=[]
        for i,item in enumerate(items):
            harmful=model.classes[q[i].argmax()]=='harmful_content'
            gate=.8776460484899311 if harmful else (0 if qc[i]>=threshold else 2)
            label=gate_label(itexts[i],model.classes,q[i],gate,sub[i]);direct.append(label is not None);output.append(fallback[item['id']] if label is None else POLICY[label].category.value)
        output=np.array(output)
        if base is None:base=output
        report={'gate':kind,'threshold':threshold,'oof_selected':len(chosen),'oof_errors':int((pred[chosen]!=gold[chosen]).sum()),'replay_direct':sum(direct),'replay_metrics':metrics(igold,output),'corrected':int(((output==igold)&(base!=igold)).sum()),'regressed':int(((output!=igold)&(base==igold)).sum())}
        reports.append(report);print(json.dumps(report),flush=True)
    out={'scope':'public_grouped_OOF_operating_point_and_frozen_input_replay_not_live','api_calls':0,'oof_hash':hashlib.sha256((cache/'jh_r8_binary_screen1/baseline.npz').read_bytes()).hexdigest(),'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Confidence gate comparison complete.\n')


if __name__=='__main__':main()
