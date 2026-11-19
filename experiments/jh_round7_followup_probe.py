"""Offline follow-up gates using a frozen public-development fallback/vector set."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from pipelines import load
load('jh_cm_char_mlp50')
from core.schema import POLICY
from pipelines.kimjaehak.classifier_models import public_data
from pipelines.kimjaehak.round7_hybrid import BlendModel
from pipelines.kimjaehak.learned_harmful import train_harmful_classifier
from experiments.jh_round7_gate_replay import gate_label
from experiments.jh_classifier_matrix import metrics


def main():
    dest=ROOT/'experiments/.cache/jh_r7_followup_probe1';dest.mkdir()
    base=ROOT/'experiments/.cache/jh_r7_hybrid_hybrid1'
    train,texts,dense,_=public_data();gold=np.array([r['labels']['category'] for r in train])
    old=json.loads((base/'manifest.json').read_text())
    if old['ids']!=[r['id'] for r in train]:raise ValueError('Training IDs mismatch')
    cp=np.load(base/'char_mlp.npz');dp=np.load(base/'dense_mlp.npz')
    assert np.array_equal(cp['classes'],dp['classes'])
    oof=np.hstack([cp['probabilities'],dp['probabilities']])
    with threadpool_limits(limits=1):
        meta=LogisticRegression(C=.1,max_iter=1000,class_weight=None).fit(np.log(np.clip(oof,1e-6,1)),gold)
    m=BlendModel().fit(texts,gold,dense)
    run=ROOT/'experiments/.cache/jh_guard_fusion_r7_g07'
    items=[json.loads(s) for s in (run/'inputs.jsonl').read_text().splitlines()]
    texts=[next(x['content'] for x in reversed(r['messages']) if x['role']=='user') for r in items]
    vectors=np.load(ROOT/'experiments/.cache/jh_r7_replay_primary1/vectors.npz')['vectors']
    fallback={r['id']:r['response']['input']['category'] for r in map(json.loads,(run/'r7_llm_input/responses.jsonl').read_text().splitlines())}
    a=m.char.probabilities(texts);b=m.dense.probabilities(texts,vectors)[:,m.dense_order]
    blend=.75*a+.25*b
    normalback=blend.copy();mask=(blend.max(1)<.8208260969823079)&(m.classes[a.argmax(1)]=='normal')&(a.max(1)>=.8556677733093049)
    normalback[mask]=a[mask]
    stack=meta.predict_proba(np.log(np.clip(np.hstack([a,b]),1e-6,1)))
    stack=stack[:,[list(meta.classes_).index(c) for c in m.classes]]
    hv,hm=train_harmful_classifier(1000);hp=hm.predict_proba(hv.transform(texts));sub=[(str(hm.classes_[p.argmax()]),float(p.max())) for p in hp]
    labels=np.array([r['labels']['category'] for r in items]);reports=[]
    for name,p,threshold in [('char',a,.8556677733093049),('blend',blend,.8208260969823079),('normalback',normalback,.8208260969823079),('stack_log01',stack,.7654121249906767)]:
        predictions=[];direct=[]
        for i,row in enumerate(items):
            label=gate_label(texts[i],m.classes,p[i],threshold,sub[i]);direct.append(label is not None)
            predictions.append(POLICY[label].category.value if label else fallback[row['id']])
        pred=np.array(predictions);direct=np.array(direct)
        report={'name':name,'metrics':metrics(labels,pred),'direct':int(direct.sum()),'direct_errors':int((pred[direct]!=labels[direct]).sum())}
        reports.append(report);print(json.dumps(report),flush=True)
        (dest/(name+'.json')).write_text(json.dumps([{'id':r['id'],'category':predictions[i],'direct':bool(direct[i])} for i,r in enumerate(items)],indent=2)+'\n')
    (dest/'summary.json').write_text(json.dumps({'scope':'public_development_frozen_fallback_probe_not_live_score','api_calls':0,'normal_backstop_count':int(mask.sum()),'results':reports},indent=2)+'\n')
    (dest/'stack_model.json').write_text(json.dumps({'classes':meta.classes_.tolist(),'coef':meta.coef_.tolist(),'intercept':meta.intercept_.tolist()},indent=2)+'\n')


if __name__=='__main__':main()
