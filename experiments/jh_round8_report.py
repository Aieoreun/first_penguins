"""Summarize completed R8 public development runs, without reading private cases."""
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.jh_neural_peace_report import safety_metrics,grounding_metrics,endpoint_runtime
from experiments.jh_compare import evaluator_hashes

DECLARED_OUTAGES = {
    'jh_guard_fusion_r8_g02': 'Timeouts/502 escalated during candidate sequence; subsequent single-call probes also failed. Preserve scores as degraded-service observations, not a healthy ranking.',
    'jh_guard_fusion_r8_g05': 'Stopped during the third candidate after severe timeouts. Complete candidate observations are separately recorded in round8_g05_outage.json; partial responses are not scored.'
}

def main():
    cache=ROOT/'experiments/.cache';runs=[]
    for prefix in ['jh_guard_fusion_r8_','jh_tutor_mlp_r8_','jh_manual_tutor_r8_']:
        for path in sorted(cache.glob(prefix+'*')):
            if not (path/('summary.json' if prefix.startswith('jh_guard_fusion') else 'COMPLETE')).exists():continue
            plan=json.loads((path/'plan.json').read_text());raw=(path/'inputs.jsonl').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=plan['dataset_sha256']:raise ValueError('Dataset changed')
            items=[json.loads(s) for s in raw.splitlines()]
            for name in plan['order']:
                r=path/name;manifest=json.loads((r/'manifest.json').read_text())
                if manifest['evaluator_hashes']!=evaluator_hashes():raise ValueError('Evaluator changed')
                rows=[json.loads(s) for s in (r/'responses.jsonl').read_text().splitlines()]
                if len(rows)!=len(items) or {x['id'] for x in rows}!={x['id'] for x in items}:raise ValueError('Response IDs differ')
                result=json.loads((r/'result.json').read_text());runtime=json.loads((r/'runtime.json').read_text())
                primary=plan.get('dataset_provenance',{}).get('primary_metric','mixed')
                endpoints={item['endpoint'] for item in items}
                score_keys=['invalid','judge_errors']
                if 'guard' in endpoints:score_keys.append('guard')
                if 'tutor' in endpoints:score_keys.append('tutor')
                if endpoints=={'guard','tutor'}:score_keys.append('total')
                entry={'run':path.name,'candidate':name,'variant':manifest['variant'],'count':len(items),
                       'dataset_sha256':plan['dataset_sha256'],'primary_metric':primary,
                       'score_scope':'+'.join(sorted(endpoints)),
                       'scores':{k:result.get(k) for k in score_keys},
                       'declared_api_outage':path.name in DECLARED_OUTAGES,
                       'comparison_note':DECLARED_OUTAGES.get(path.name,'Public development run; inspect internal errors and repeat variability before attributing differences.'),
                       'runtime':runtime,'endpoint_runtime':endpoint_runtime(rows,(r/'server.log').read_text()),
                       'guard_details':result.get('guard_details'),'safety':safety_metrics(items,rows),
                       'source_commit':plan.get('current_commit',plan.get('commit'))}
                if primary not in ['safe_accuracy','input_only']:entry['grounding']=grounding_metrics(items,rows)
                if primary=='safe_accuracy':entry['scores']={k:result.get(k) for k in ['invalid','judge_errors']}
                runs.append(entry)
    report={'reporting_version':2,'scope':'completed_R8_public_development_not_private_submission',
            'score_scope_note':'Guard-only/tutor-only runs omit the other endpoint and misleading half-weight aggregate; raw official scorer results remain in the source cache.',
            'runs':runs,'candidate_runs':len(runs),'repeated_responses':sum(r['count'] for r in runs)}
    (ROOT/'docs/jh/round8_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in runs:print(r['run'],r['candidate'],r['scores'],r['runtime']['collect_seconds'])
if __name__=='__main__':main()
