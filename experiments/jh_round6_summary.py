"""Summarize completed public/development R6 runs; never rejudge responses."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summary():
    done, pending = [], []
    for manifest_path in sorted((ROOT/'experiments/.cache').glob('jh_*r6_*/*/manifest.json')):
        run = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        result_path = run/'derived_result.json'
        if not result_path.exists():
            result_path = run/'result.json'
        if not result_path.exists():
            pending.append(str(run.relative_to(ROOT)))
            continue
        result = json.loads(result_path.read_text())
        runtime = json.loads((run/'runtime.json').read_text())
        if (run.parent/'derived_inputs.jsonl').exists():
            items = list(map(json.loads,(run.parent/'derived_inputs.jsonl').read_text().splitlines()))
        else:
            items = [row for name in ('tutor','guard') for row in
                     map(json.loads,(ROOT/'public'/f'{name}.jsonl').read_text().splitlines())]
        responses = list(map(json.loads,(run/'responses.jsonl').read_text().splitlines()))
        by_id = {r['id']:r['response'] or {} for r in responses}
        assert len(responses) == len(by_id) == manifest['count']
        items = [i for i in items if i['id'] in by_id]
        normal = [i for i in items if i.get('endpoint','tutor') == 'tutor' and i['labels']['category'] == 'normal']
        kind = manifest.get('diagnostic_kind') or manifest.get('paraphrase_partition') or 'public'
        done.append({'run':str(run.relative_to(ROOT)), 'variant':manifest['variant'],
            'dataset':kind, 'count':manifest['count'], 'commit':manifest['commit'], 'dirty':manifest['dirty'],
            'dataset_sha256':manifest.get('dataset_sha256'), 'evaluator_hashes':manifest['evaluator_hashes'],
            'tutor':result['tutor'], 'guard':result['guard'], 'invalid':result['invalid'],
            'judge_errors':result['judge_errors'], 'runtime':runtime, 'per_category':result['per_category'],
            'normal_count':len(normal), 'normal_actions':dict(Counter(by_id[i['id']].get('action') for i in normal)),
            'normal_citation_hits':sum(bool(set(i['labels'].get('evidence',[])) &
                                           set(by_id[i['id']].get('citations',[]))) for i in normal),
            'embedding_index_loaded':manifest.get('embedding_index_loaded'),
            'effective_config':{k:v for k,v in manifest.get('effective_config',{}).items()
                                if k in {'class','request_slots','embedding_mode','embedding_model',
                                         'tutor_short_material','tutor_material_no_section','tutor_verify_context_sources',
                                         'source_evidence_order','source_selection_votes','tutor_computable_material','tutor_topic_selection','source_resolved_hint','source_resolved_hint_prompt_sha256','source_selection_effort','source_selection_tokens','source_extract_timeout','tutor_normal_gate_threshold','tutor_joint_evidence','joint_evidence_tokens','joint_evidence_prompt_sha256','tutor_skip_trusted_embedding','tutor_lexical_mode','tutor_resolved_query','query_resolve_effort',
                                         'source_extract_sections','tutor_prepare_limit','tutor_material_timeout'}}})
    return {'scope':'public_development_not_private_score','completed_runs':len(done),
            'total_requests':sum(r['count'] for r in done), 'pending_runs':pending, 'runs':done}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summary()
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for row in result['runs']:
        print(row['run'].split('/')[-2], row['variant'],row['tutor'],row['runtime']['collect_seconds'],
              row['runtime']['stage_failure_requests'], row['normal_actions'])
    print('COMPLETED',result['completed_runs'],'PENDING',len(result['pending_runs']))
