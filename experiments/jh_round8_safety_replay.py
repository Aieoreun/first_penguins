"""Replay safety routing on fixed public outputs; not HTTP or independent gold."""
import hashlib,json
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from pipelines.kimjaehak.classifier_models import CategoryModel
from pipelines.kimjaehak.span_tagger import public_rows
from experiments.jh_guard_fusion_datasets import load_dataset
from experiments.jh_neural_peace_report import safety_metrics
from experiments.jh_compare import score_rows

def main():
    dest=ROOT/'experiments/.cache/jh_r8_safety_replay1';dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE').exists():raise ValueError('Already completed')
    rows=public_rows();texts=[r['candidate_answer'] for r in rows];labels=['safe' if r['labels']['safe'] else 'unsafe' for r in rows]
    model=CategoryModel('char_lr',c=100).fit(texts,labels);index=int(np.flatnonzero(model.classes=='unsafe')[0]);reports=[]
    for dataset in ['paraphrase','novel','cross_safety','refusal']:
        items,provenance=load_dataset(dataset,ROOT);selected=[r for r in items if r.get('candidate_answer') is not None]
        p=model.probabilities([r['candidate_answer'] for r in selected])[:,index]
        result={'dataset':dataset,'provenance':provenance,'count':len(selected),'unsafe':sum(r['labels']['safe'] is False for r in selected),'thresholds':[]}
        for threshold in [.1,.5]:
            routed=p>=threshold;unsafe=np.array([r['labels']['safe'] is False for r in selected]);result['thresholds'].append({'threshold':threshold,'routed':int(routed.sum()),'unsafe_not_routed':int((unsafe&~routed).sum()),'safe_routed':int((~unsafe&routed).sum())})
        reports.append(result)
    run=ROOT/'experiments/.cache/jh_guard_fusion_r8_g01';items=[json.loads(x) for x in (run/'inputs.jsonl').read_text().splitlines()]
    baseline=[json.loads(x) for x in (run/'r7_binary_guard/responses.jsonl').read_text().splitlines()];independent={r['id']:r for r in map(json.loads,(run/'r8_binary_parallel/responses.jsonl').read_text().splitlines())};assert {r['id'] for r in baseline}==set(independent)=={r['id'] for r in items}
    inputs={r['id']:r for r in items};replays=[]
    for threshold in [None,.1,.5]:
        output=deepcopy(baseline);changed=routed=0
        for row in output:
            item=inputs[row['id']]
            if threshold is None or item.get('candidate_answer') is None:continue
            risk=model.probabilities([item['candidate_answer']])[0,index]
            if risk>=threshold:
                routed+=1;value=independent[row['id']]['response']['output']['safe'];changed+=row['response']['output']['safe']!=value;row['response']['output']['safe']=value
        score=score_rows(items,output,1,no_judge=True)
        replays.append({'threshold':threshold,'routed':routed,'changed_safety':changed,'replay_guard':score['guard'],'invalid':score['invalid'],'safety':safety_metrics(items,output)})
    report={'scope':'public_output_routing_diagnostic_and_frozen_safety_replay_not_live_not_holdout','api_calls':0,'limitations':['Public candidate texts overlap training; novel paraphrases are not an independent holdout.','GPT results reused from prior independent check; no new output judgments or real latency measured.','Guard total is a counterfactual replay, not a submitted/full-server score.'],'routing':reports,'replay':replays,'model_fits':model.diagnostics}
    (dest/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(dest/'COMPLETE').write_text('Offline safety replay complete.\n')
    for r in reports:print(r['dataset'],r['thresholds'])
    for r in replays:print('replay',r['threshold'],r['routed'],r['changed_safety'],r['replay_guard'])
if __name__=='__main__':main()
