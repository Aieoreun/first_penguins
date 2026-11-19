"""Read-only analysis of guard responses; never imported by production pipelines."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.score_local import as_scoring_input
from grader.scoring import parse_guard


def safety_metrics(items, rows):
    responses = {r['id']:r['response'] for r in rows}
    groups = {key:Counter() for key in ('all','gold_allow','gold_block','gold_normal')}
    failures=[]
    for row in items:
        if row['endpoint']!='guard' or 'candidate_answer' not in row: continue
        item,variant=as_scoring_input(row)
        gold=variant['safe']
        response=parse_guard(responses[row['id']],True)
        pred=response['output']['safe'] if response else None
        category=item.get('category')
        selected=['all','gold_'+item['labels']['decision']]
        if category=='normal': selected.append('gold_normal')
        for key in selected:
            c=groups[key]
            c['count']+=1
            c['gold_safe' if gold else 'gold_unsafe']+=1
            if pred is None: c['invalid']+=1
            elif pred==gold: c['correct']+=1
            elif gold: c['false_alarm_safe_as_unsafe']+=1
            else: c['miss_unsafe_as_safe']+=1
        if pred!=gold: failures.append({'id':row['id'],'gold_safe':gold,'predicted_safe':pred})
    return {'groups':{k:dict(v) for k,v in groups.items()},'errors':failures,
            'definition':'Positive means unsafe: false_alarm=safe->unsafe, miss=unsafe->safe; invalid is separate.'}


def grounding_metrics(items, rows):
    """Expose error direction and span F1 on ungrounded outputs, not only easy empties."""
    from grader.scoring import span_f1
    responses={r['id']:r['response'] for r in rows}
    result={}
    for group in ('all','gold_allow'):
        counts=Counter(); spans=[]
        for row in items:
            if row['endpoint']!='guard' or 'candidate_answer' not in row: continue
            if group=='gold_allow' and row['labels']['decision']!='allow': continue
            item,variant=as_scoring_input(row)
            gold=item['labels']['output_label']!='ungrounded'
            parsed=parse_guard(responses[row['id']],True)
            counts['count']+=1
            counts['gold_grounded' if gold else 'gold_ungrounded']+=1
            if parsed is None:
                counts['invalid']+=1
                if not gold:spans.append(0)
                continue
            pred=parsed['output']['grounded']
            if pred==gold:counts['correct']+=1
            elif gold:counts['false_alarm_grounded_as_ungrounded']+=1
            else:counts['miss_ungrounded_as_grounded']+=1
            if not gold:spans.append(span_f1(parsed['output']['unsupported'],variant['unsupported_spans']))
        result[group]={**counts,'ungrounded_span_f1':sum(spans)/len(spans) if spans else None}
    return result


def endpoint_runtime(rows, log):
    import re
    import statistics
    result={}
    lines=re.findall(r'\b(tutor|guard) [^\n]* errors=([^\s]+)',log)
    for endpoint in sorted({row['endpoint'] for row in rows}):
        durations=sorted(row['seconds'] for row in rows if row['endpoint']==endpoint)
        errors=[error for name,error in lines if name==endpoint]
        result[endpoint]={'count':len(durations),'p50':statistics.median(durations),
            'p95':durations[min(len(durations)-1,int(.95*len(durations)))],'max':max(durations),
            'logged_requests':len(errors),'stage_failure_requests':sum(e!='-' for e in errors),
            'timeout_requests':sum(any(word in e for word in ('Timeout','Deadline','NoBudget','tie_budget')) for e in errors),
            'stage_errors':dict(Counter(part for e in errors if e!='-' for part in e.split(',')))}
    return result


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from experiments import jh_guard_fusion_compare as runner
    ap=argparse.ArgumentParser(description='Aggregate completed neural/Peace runs without API calls')
    ap.add_argument('--guard-runs',nargs='+',required=True)
    ap.add_argument('--mixed-runs',nargs='*',default=[])
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    report={'scope':'local_public_derived_development_not_private_submission','runs':[],
            'private_evidence':{'own_baseline':{'variant':'jh_cm_char_mlp50','total':85.41,'tutor':81.25,'guard':89.57},
                'peace':{'commit':'3c64079eb1af14170b3bee4578ac6f916bec433e','guard_over_90_user_confirmed':True,
                         'exact_private_score':None,'public_mean_90_52_is_separate':True}}}
    for kind,names in (('guard',args.guard_runs),('mixed',args.mixed_runs)):
        for name in names:
            prefix='jh_guard_fusion_' if kind=='guard' else 'jh_tutor_mlp_'
            root=runner.ROOT/'experiments/.cache'/(prefix+name)
            plan=json.loads((root/'plan.json').read_text())
            items=[json.loads(s) for s in (root/'inputs.jsonl').read_text().splitlines()]
            if hashlib.sha256((root/'inputs.jsonl').read_bytes()).hexdigest()!=plan['dataset_sha256']:
                raise ValueError('Frozen dataset changed')
            for label in plan['order']:
                path=root/label
                manifest=json.loads((path/'manifest.json').read_text())
                if manifest['evaluator_hashes']!=runner.shared.evaluator_hashes():raise ValueError('Evaluator changed')
                rows=[json.loads(s) for s in (path/'responses.jsonl').read_text().splitlines()]
                if len(rows)!=len(items) or {x['id'] for x in rows}!={x['id'] for x in items}:raise ValueError('Incomplete responses')
                result=json.loads((path/'result.json').read_text())
                primary=plan.get('dataset_provenance',{}).get('primary_metric','mixed')
                entry={'run':name,'kind':kind,'candidate':label,'variant':manifest['variant'],
                    'count':len(rows),'dataset_sha256':plan['dataset_sha256'],
                    'primary_metric':primary,'scores':{k:result.get(k) for k in ('total','tutor','guard','invalid','judge_errors')},
                    'guard_details':result['guard_details'],'safety':safety_metrics(items,rows),
                    'runtime':json.loads((path/'runtime.json').read_text()),
                    'endpoint_runtime':endpoint_runtime(rows,(path/'server.log').read_text()),
                    'effective_config':manifest['effective_config'],'source_commit':plan.get('current_commit',plan.get('commit')),
                    'cache':str(path.relative_to(runner.ROOT))}
                if primary not in ('safe_accuracy','input_only'):
                    entry['grounding']=grounding_metrics(items,rows)
                if primary in ('safe_accuracy','input_only'):
                    # Do not publish non-primary synthetic aggregate scores.
                    entry['scores']={k:result.get(k) for k in ('invalid','judge_errors')}
                    entry['guard_details']={k:result['guard_details'][k] for k in
                        ('block_f1','category_accuracy','false_block','missed_block','confusion','safe_accuracy')}
                if 'intent_details' in result:entry['intent']=result['intent_details']
                if 'relational_details' in result:entry['paired_grounding']=result['relational_details']
                if 'counterfactual_details' in result:entry['paired_grounding']=result['counterfactual_details']
                report['runs'].append(entry)
    report['requests']=sum(x['count'] for x in report['runs'])
    report['runs_count']=len(report['runs'])
    runner.write_json(Path(args.output),report)
    print(json.dumps({'runs':report['runs_count'],'requests':report['requests'],'output':args.output}))


if __name__=='__main__':main()
