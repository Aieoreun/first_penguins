"""Check history-aware abstention on a frozen public GPT input fallback."""
import hashlib,json,re
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.classifier_models import CategoryModel,public_data
from pipelines.kimjaehak.round7_binary_model import BinaryAdjustedModel,BLOCKED
from experiments.jh_classifier_matrix import metrics

REFERENCE=re.compile(r'그거|그걸|그것|그대로|이거|이걸|방금|앞서|위에서|그\s*방법|계속|이어서|그럼')


def main():
    cache=ROOT/'experiments/.cache';dest=cache/'jh_r8_history_replay1';dest.mkdir()
    source=cache/'jh_guard_fusion_r7_g07';raw=(source/'inputs.jsonl').read_bytes()
    items=[json.loads(x) for x in raw.splitlines()]
    prior=json.loads((cache/'jh_r8_cross_replay1/results.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==prior['dataset_sha256']
    baseline_rows=json.loads((cache/'jh_r8_cross_replay1/baseline.json').read_text())
    assert [r['id'] for r in baseline_rows]==[r['id'] for r in items]
    fallback_path=source/'r7_llm_input/responses.jsonl'
    assert hashlib.sha256(fallback_path.read_bytes()).hexdigest()==prior['fallback_sha256']
    fallback={r['id']:r['response']['input']['category'] for r in map(json.loads,fallback_path.read_text().splitlines())}
    rows,texts,_,_=public_data();labels=[r['labels']['category'] for r in rows]
    category=CategoryModel('char_mlp').fit(texts,labels)
    binary=CategoryModel('char_mlp').fit(texts,['block' if c in BLOCKED else 'allow' for c in labels])
    model=BinaryAdjustedModel(category,binary)
    contexts=['\n'.join(m['content'] for m in r['messages'] if m['role']=='user')[-2400:] for r in items]
    hp=model.probabilities(contexts);hc=model.classes[hp.argmax(1)]
    gold=np.array([r['labels']['category'] for r in items]);base=np.array([r['category'] for r in baseline_rows])
    direct=np.array([r['direct'] for r in baseline_rows]);reports=[]
    for mode in ['baseline','any_history','short_reference','group_disagreement','category_disagreement']:
        defer=[]
        for i,item in enumerate(items):
            users=[m['content'] for m in item['messages'] if m['role']=='user'];multi=len(users)>1
            flag=(multi if mode=='any_history' else
                  multi and len(users[-1])<=80 and bool(REFERENCE.search(users[-1])) if mode=='short_reference' else
                  multi and ((hc[i] in BLOCKED)!=(base[i] in BLOCKED)) if mode=='group_disagreement' else
                  multi and hc[i]!=base[i] if mode=='category_disagreement' else False)
            defer.append(flag and direct[i])
        defer=np.array(defer);pred=base.copy()
        for i in np.flatnonzero(defer):pred[i]=fallback[items[i]['id']]
        report={'mode':mode,'metrics':metrics(gold,pred),'extra_GPT_input_requests':int(defer.sum()),
                'corrected':int(((pred==gold)&(base!=gold)).sum()),
                'regressed':int(((pred!=gold)&(base==gold)).sum()),
                'changed_ids':[items[i]['id'] for i in np.flatnonzero(pred!=base)]}
        reports.append(report);print(json.dumps(report),flush=True)
    out={'scope':'public_frozen_fallback_history_abstention_not_live_or_holdout','api_calls':0,
         'dataset_sha256':prior['dataset_sha256'],'fallback_sha256':prior['fallback_sha256'],
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('History abstention replay complete.\n')


if __name__=='__main__':main()
