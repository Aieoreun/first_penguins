"""Aggregate completed reasoning experiments, without generating or judging again."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import jh_guard_fusion_compare as runner
from experiments.jh_neural_peace_report import endpoint_runtime, grounding_metrics, safety_metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--guard-runs',nargs='*',default=[])
    ap.add_argument('--tutor-runs',nargs='*',default=[])
    ap.add_argument('--mixed-runs',nargs='*',default=[])
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    report={'scope':'public_derived_reasoning_development_not_private_submission','runs':[]}
    for kind,names in (('guard',args.guard_runs),('tutor',args.tutor_runs),('mixed',args.mixed_runs)):
        for name in names:
            prefix='jh_guard_fusion_' if kind=='guard' else 'jh_tutor_mlp_'
            root=ROOT/'experiments/.cache'/(prefix+name)
            plan=json.loads((root/'plan.json').read_text())
            raw=(root/'inputs.jsonl').read_bytes()
            if runner.digest(raw)!=plan['dataset_sha256']:raise ValueError('Dataset changed')
            items=[json.loads(s) for s in raw.splitlines()]
            for label in plan['order']:
                path=root/label
                manifest=json.loads((path/'manifest.json').read_text())
                if manifest['evaluator_hashes']!=runner.shared.evaluator_hashes():raise ValueError('Evaluator changed')
                rows=[json.loads(s) for s in (path/'responses.jsonl').read_text().splitlines()]
                if len(rows)!=len(items) or {x['id'] for x in rows}!={x['id'] for x in items}:raise ValueError('Incomplete collection')
                result=json.loads((path/'result.json').read_text())
                runtime=json.loads((path/'runtime.json').read_text())
                endpoints=manifest['endpoints']
                scores={key:result[key] for key in endpoints+['invalid','judge_errors']}
                if len(endpoints)==2:scores['total']=result['total']
                entry={'run':name,'kind':kind,'candidate':label,'variant':manifest['variant'],
                    'count':len(rows),'dataset_sha256':plan['dataset_sha256'],
                    'scope':manifest.get('scope'), 'score_scope':manifest['score_scope'],
                    'scores':scores,'runtime':runtime,
                    'endpoint_runtime':endpoint_runtime(rows,(path/'server.log').read_text()),
                    'effective_config':manifest['effective_config'],
                    'source_commit':plan.get('current_commit',plan.get('commit')),
                    'cache':str(path.relative_to(ROOT)),
                    'medium_calls':runtime['api_calls'].get('chat_efforts',{}).get('medium',0)}
                if 'guard' in endpoints:
                    entry['guard_details']=result['guard_details']
                    entry['safety']=safety_metrics(items,rows)
                    entry['grounding']=grounding_metrics(items,rows)
                report['runs'].append(entry)
    report['requests']=sum(x['count'] for x in report['runs'])
    report['runs_count']=len(report['runs'])
    summary=[]
    for scope,dataset,candidate in sorted({(x['score_scope'],x['dataset_sha256'],x['candidate']) for x in report['runs']}):
        selected=[x for x in report['runs'] if x['score_scope']==scope
                  and x['dataset_sha256']==dataset and x['candidate']==candidate]
        values={}
        for metric in ('tutor','guard','total'):
            points=[x['scores'][metric] for x in selected if metric in x['scores']]
            if points:values[metric]={'mean':statistics.mean(points),'min':min(points),'max':max(points),
                'stdev':statistics.stdev(points) if len(points)>1 else None}
        summary.append({'scope':scope,'dataset_sha256':dataset,'candidate':candidate,
                        'repetitions':len(selected),'scores':values})
    report['summary']=summary
    runner.write_json(Path(args.output),report)
    print(json.dumps({'runs':report['runs_count'],'requests':report['requests'],'output':args.output}))


if __name__=='__main__':main()
