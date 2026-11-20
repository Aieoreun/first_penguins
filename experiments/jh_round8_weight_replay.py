"""Replay softened class balancing against the same frozen GPT input fallback."""
import hashlib,json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipelines import load
load('jh_r7_binary_guard')
from core.schema import POLICY
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_binary_model import BinaryAdjustedModel,BLOCKED
from pipelines.kimjaehak.learned_harmful import train_harmful_classifier
from experiments.jh_round7_gate_replay import gate_label
from experiments.jh_classifier_matrix import metrics


def main():
    cache=ROOT/'experiments/.cache';dest=cache/'jh_r8_weight_replay1';dest.mkdir()
    source=cache/'jh_guard_fusion_r7_g07'
    raw=(source/'inputs.jsonl').read_bytes();items=[json.loads(s) for s in raw.splitlines()]
    assert hashlib.sha256(raw).hexdigest()==json.loads((source/'plan.json').read_text())['dataset_sha256']
    fallback_path=source/'r7_llm_input/responses.jsonl'
    fallback={r['id']:r['response']['input']['category'] for r in map(json.loads,fallback_path.read_text().splitlines())}
    assert set(fallback)=={r['id'] for r in items}
    texts=[next(m['content'] for m in reversed(r['messages']) if m['role']=='user') for r in items]
    gold=np.array([r['labels']['category'] for r in items])
    train,tt,_,_=public_data();labels=[r['labels']['category'] for r in train]
    binary=CategoryModel('char_mlp').fit(tt,['block' if c in BLOCKED else 'allow' for c in labels])
    hv,hm=train_harmful_classifier(1000);sub=[]
    for p in hm.predict_proba(hv.transform(texts)):sub.append((str(hm.classes_[p.argmax()]),float(p.max())))
    cv=json.loads((cache/'jh_r8_weight_screen1/results.json').read_text())
    point=next(r for r in cv['results'] if r['category_power']==.5 and r['binary_power']==1.)
    thresholds={'baseline':.8776460484899311,'soft_category':point['matched_coverage']['0.5']['min_confidence']}
    reports=[];baseline=None
    for name in thresholds:
        weights=compute_sample_weight('balanced',labels)
        if name=='soft_category':weights=np.sqrt(weights);weights=weights/weights.mean()
        with patch('pipelines.kimjaehak.classifier_models.compute_sample_weight',return_value=weights):
            category=CategoryModel('char_mlp').fit(tt,labels)
        model=BinaryAdjustedModel(category,binary);prob=model.probabilities(texts)
        pred=[];direct=[]
        for i,item in enumerate(items):
            label=gate_label(texts[i],model.classes,prob[i],thresholds[name],sub[i]);direct.append(label is not None)
            pred.append(fallback[item['id']] if label is None else POLICY[label].category.value)
        pred=np.array(pred);direct=np.array(direct)
        if baseline is None:
            baseline=pred
            previous=json.loads((cache/'jh_r8_cross_replay1/baseline.json').read_text())
            assert pred.tolist()==[r['category'] for r in previous]
        report={'candidate':name,'threshold':thresholds[name],'metrics':metrics(gold,pred),
                'direct':int(direct.sum()),'direct_errors':int((pred[direct]!=gold[direct]).sum()),
                'corrected':int(((pred==gold)&(baseline!=gold)).sum()),
                'regressed':int(((pred!=gold)&(baseline==gold)).sum())}
        reports.append(report);print(json.dumps(report),flush=True)
        (dest/(name+'.json')).write_text(json.dumps([{'id':r['id'],'category':str(pred[i]),'direct':bool(direct[i])} for i,r in enumerate(items)],indent=2)+'\n')
    out={'scope':'frozen_GPT_input_replay_public_development_not_holdout','api_calls':0,
         'dataset_sha256':hashlib.sha256(raw).hexdigest(),
         'fallback_sha256':hashlib.sha256(fallback_path.read_bytes()).hexdigest(),
         'cv_sha256':hashlib.sha256((cache/'jh_r8_weight_screen1/results.json').read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Weight replay completed.\n')


if __name__=='__main__':main()
