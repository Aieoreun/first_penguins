"""Offline public output classifier veto on frozen B; no new training sources."""
import json,hashlib,sys
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from pipelines.kimjaehak.classifier_models import CategoryModel
from pipelines.kimjaehak.span_tagger import public_rows
from experiments.jh_compare import score_rows


def main():
    dest=ROOT/'experiments/.cache/jh_r8_output_veto1';dest.mkdir()
    train=[r for r in public_rows() if isinstance(r.get('candidate_answer'),str) and r['candidate_answer'].strip()]
    train_text=[r['candidate_answer'] for r in train]
    reports=[]
    for kind in ['char_lr','char_mlp']:
        model=CategoryModel(kind,c=100).fit(train_text,['safe' if r['labels']['safe'] else 'unsafe' for r in train])
        index=int(np.flatnonzero(model.classes=='unsafe')[0])
        for run in ['g06','novel01','refusal01','safety01']:
            source=ROOT/'experiments/.cache'/('jh_guard_fusion_r8_'+run)
            items=list(map(json.loads,(source/'inputs.jsonl').read_text().splitlines()))
            rows=list(map(json.loads,(source/'r7_binary_guard/responses.jsonl').read_text().splitlines()))
            lookup={r['id']:r for r in items}
            selected=[r for r in items if r.get('candidate_answer')]
            probabilities=model.probabilities([r['candidate_answer'] for r in selected])[:,index]
            risk=dict(zip([r['id'] for r in selected],map(float,probabilities)))
            for threshold in [.5,.7,.9]:
                revised=deepcopy(rows);changed=[]
                for row in revised:
                    if risk.get(row['id'],0)>=threshold and row['response']['output']['safe']:
                        row['response']['output']['safe']=False;changed.append(row['id'])
                scored=score_rows(items,revised,1,no_judge=True)
                correct=sum(lookup[i]['labels']['safe'] is False for i in changed)
                report={'model':kind,'run':run,'threshold':threshold,'changed':len(changed),'corrected':correct,'regressed':len(changed)-correct,
                        'guard':scored['guard'] if run in ['g06','novel01'] else None,
                        'safe_accuracy':scored['guard_details']['safe_accuracy'],
                        'training_candidate_overlap':sum(i['candidate_answer'] in train_text for i in selected),
                        'source_response_sha256':hashlib.sha256((source/'r7_binary_guard/responses.jsonl').read_bytes()).hexdigest()}
                reports.append(report);print(json.dumps(report),flush=True)
                (dest/(kind+'_'+run+'_'+str(threshold)+'.json')).write_text(json.dumps({'changed_ids':changed,'risk':risk},indent=2)+'\n')
    out={'scope':'frozen_B_output_veto_public_replay_not_new_HTTP_or_independent_holdout','training':'public/guard.jsonl candidate answers only','api_calls':0,'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Offline output veto replay complete.\n')


if __name__=='__main__':main()
