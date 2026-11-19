"""Read-only output comparison where two recorded guard inputs agree.

Conditioning on predicted ALLOW is a descriptive subset, not causal matching or
an independent holdout. Full-set official scores remain the primary results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.jh_neural_peace_report import grounding_metrics,safety_metrics
from experiments.jh_compare import evaluator_hashes


def compare(path,baseline,candidate):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT/'experiments/.cache'):
        raise ValueError('Only recorded local development runs')
    plan=json.loads((path/'plan.json').read_text())
    raw=(path/'inputs.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=plan['dataset_sha256']:
        raise ValueError('Changed development inputs')
    items=[json.loads(s) for s in raw.splitlines()]
    ids={x['id'] for x in items};data=[]
    for label in (baseline,candidate):
        if label not in plan['order']:raise ValueError('Candidate not in frozen plan')
        root=path/label
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['evaluator_hashes']!=evaluator_hashes():raise ValueError('Evaluator changed')
        result=json.loads((root/'result.json').read_text())
        if result['invalid'] or result['judge_errors']:raise ValueError('Incomplete valid scoring')
        rows=[json.loads(s) for s in (root/'responses.jsonl').read_text().splitlines()]
        if len(rows)!=len(items) or {r['id'] for r in rows}!=ids:raise ValueError('Incomplete responses')
        data.append({r['id']:r for r in rows})
    a,b=data
    selected=[i for i in items if i['endpoint']=='guard' and i.get('candidate_answer') is not None
        and a[i['id']]['response']['input']==b[i['id']]['response']['input']
        and a[i['id']]['response']['input']['decision']=='allow']
    changed={key:0 for key in ('grounded','unsupported','safe','category')}
    for item in selected:
        x,y=[d[item['id']]['response']['output'] for d in data]
        for key in changed:changed[key]+=x.get(key)!=y.get(key)
    return {'run':path.name,'baseline':baseline,'candidate':candidate,
        'dataset_sha256':plan['dataset_sha256'],'matched_allow_count':len(selected),
        'changed':changed,'baseline_output':{'grounding':grounding_metrics(selected,list(a.values())),
            'safety':safety_metrics(selected,list(a.values()))['groups']},
        'candidate_output':{'grounding':grounding_metrics(selected,list(b.values())),
            'safety':safety_metrics(selected,list(b.values()))['groups']}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='append',required=True,help='cache directory:baseline:candidate')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    entries=[]
    for spec in args.run:
        directory,baseline,candidate=spec.split(':')
        entries.append(compare(ROOT/'experiments/.cache'/directory,baseline,candidate))
    report={'scope':'recorded_predicted_allow_agreement_subset_not_private_not_causal',
        'api_calls':0,'limitations':['Selection depends on two model predictions.',
            'Excludes input differences and BLOCK outputs; not a replacement for full-set scores.',
            'Previously used development data; cannot estimate independent generalization.'],
        'runs':entries}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in entries:
        print(json.dumps({k:r[k] for k in ('run','baseline','candidate','matched_allow_count','changed')}))


if __name__=='__main__':main()
