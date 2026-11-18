"""Aggregate existing public-development results; no API calls or relabeling."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(run):
    plan=json.loads((run/'plan.json').read_text())
    items=[json.loads(s) for s in (run/'inputs.jsonl').read_text().splitlines()]
    summaries=[];responses={};points={}
    for label in plan['order']:
        d=run/label
        result=json.loads((d/'result.json').read_text())
        rows={r['id']:r for r in map(json.loads,(d/'responses.jsonl').read_text().splitlines())}
        scores={r['id']:r['points'] for r in result['items']}
        groups={}
        for item in items:
            if item['endpoint']!='tutor':continue
            response=rows[item['id']]['response'] or {}
            for group in ['all', 'single_turn' if len(item['messages'])==1 else 'multi_turn',item['labels']['category']]:
                bucket=groups.setdefault(group,{'count':0,'action_correct':0,'points_sum':0})
                bucket['count']+=1
                bucket['action_correct']+=response.get('action')==item['labels']['action']
                bucket['points_sum']+=scores[item['id']]
        for bucket in groups.values():
            bucket['mean_item_points']=bucket.pop('points_sum')/bucket['count']
            bucket['action_accuracy']=bucket['action_correct']/bucket['count']
        summaries.append({'candidate':label,**{k:result[k] for k in ('total','tutor','guard','invalid','judge_errors')},
            'guard_details':result['guard_details'],'tutor_groups':groups,
            'runtime':json.loads((d/'runtime.json').read_text()),
            'effective_config':json.loads((d/'startup_identity.json').read_text())})
        responses[label]=rows;points[label]=scores
    baseline='cm_char_mlp50';paired={}
    if baseline in responses:
        ids=[i['id'] for i in items if i['endpoint']=='tutor']
        for label in responses:
            if label==baseline:continue
            same=[i for i in ids if responses[label][i]['response']==responses[baseline][i]['response']]
            paired[label]={'tutor_count':len(ids),'higher_item_score':sum(points[label][i]>points[baseline][i] for i in ids),
                'lower_item_score':sum(points[label][i]<points[baseline][i] for i in ids),
                'same_response':len(same),'same_response_different_judge_score':sum(points[label][i]!=points[baseline][i] for i in same)}
    return {'name':run.name,'plan':plan,'summaries':summaries,'paired':paired}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('names',nargs='+');args=ap.parse_args()
    reports=[]
    for name in args.names:
        if not name.replace('_','').replace('-','').isalnum():ap.error('Invalid run name')
        reports.append(summarize(ROOT/'experiments/.cache'/('jh_tutor_mlp_'+name)))
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
