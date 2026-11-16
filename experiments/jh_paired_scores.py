"""Paired descriptive uncertainty for already-scored development runs, no APIs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(baseline, candidate, repetitions=10000):
    import numpy as np
    runs=[path.resolve() for path in (baseline,candidate)]
    if any(not p.is_relative_to(ROOT/'experiments/.cache') for p in runs):
        raise ValueError('Expected local development runs')
    data=[]
    for run in runs:
        inputs=run.parent/'derived_inputs.jsonl'
        result=run/'derived_result.json'
        rows=list(map(json.loads,inputs.read_text().splitlines()))
        scored=json.loads(result.read_text())
        if scored.get('judge_errors'):
            raise ValueError('Judge errors make paired aggregates incomparable')
        points={r['id']:r['points'] for r in scored['items'] if r['endpoint']=='tutor'}
        if len(points)!=len(rows) or set(points)!={r['id'] for r in rows}:
            raise ValueError('Expected complete tutor-only derived scores')
        if not points or abs(sum(points.values())/len(points)-scored['tutor']) > .011:
            raise ValueError('Item points do not reproduce tutor score')
        data.append((rows,points,hashlib.sha256(inputs.read_bytes()).hexdigest(),scored))
    if data[0][2]!=data[1][2]:
        raise ValueError('Datasets differ')
    rows,a,_,old=data[0];b=data[1][1]
    deltas={i:b[i]-a[i] for i in a}
    grouped={}
    for row in rows:
        # Two dialogue transforms of one source question stay in one cluster.
        key=row.get('source_id',row['id'])
        grouped.setdefault(key,[]).append(deltas[row['id']])
    totals=np.array([sum(v) for v in grouped.values()]);sizes=np.array([len(v) for v in grouped.values()])
    rng=np.random.default_rng(20261116)
    draws=rng.integers(0,len(grouped),size=(repetitions,len(grouped)))
    sampled=totals[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    ci=np.quantile(sampled,[.025,.975]).tolist()
    return {'scope':'observed_development_pairs_not_unseen_or_judge_variance',
            'baseline':str(runs[0].relative_to(ROOT)),'candidate':str(runs[1].relative_to(ROOT)),
            'dataset_sha256':data[0][2], 'count':len(rows),'clusters':len(grouped),
            'baseline_tutor':old['tutor'],'candidate_tutor':data[1][3]['tutor'],
            'mean_paired_delta':sum(deltas.values())/len(deltas),'bootstrap95':ci,
            'wins':sum(x>0 for x in deltas.values()),'losses':sum(x<0 for x in deltas.values()),
            'ties':sum(x==0 for x in deltas.values()),'bootstrap_repetitions':repetitions,
            'limitations':['Already used for development; not an independent holdout.',
                           'Conditions on one pair of responses/judgments; does not estimate repeat-run variance.']}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline',type=Path);ap.add_argument('candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'experiments/.cache') or output.exists():
        ap.error('Use a new local diagnostic output')
    result=compare(args.baseline,args.candidate)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
