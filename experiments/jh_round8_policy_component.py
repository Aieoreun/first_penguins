"""Freeze public B routes and change only harmful tutor wording, not the judge."""
import argparse,asyncio,hashlib,json,shutil,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.jh_refusal_generation_probe import predicted_harmful
from experiments.jh_compare import score_rows,evaluator_hashes
from experiments.jh_environment import load_environment
from pipelines import load


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-run',required=True);ap.add_argument('--name',required=True);args=ap.parse_args()
    if any(not s.replace('_','').isalnum() for s in [args.source_run,args.name]):ap.error('Simple cache names required')
    source=ROOT/'experiments/.cache'/('jh_tutor_mlp_'+args.source_run);base=source/'r7_binary_guard'
    if not (source/'COMPLETE').exists():raise ValueError('Completed public source required')
    plan=json.loads((source/'plan.json').read_text());manifest=json.loads((base/'manifest.json').read_text());original=json.loads((base/'result.json').read_text())
    assert plan['scope']=='public_derived_tutor_mlp_development_not_independent_holdout'
    assert sha(source/'inputs.jsonl')==plan['dataset_sha256'] and manifest['variant']=='jh_r7_binary_guard'
    assert manifest['evaluator_hashes']==evaluator_hashes() and not original['judge_errors'] and not original['invalid']
    items=list(map(json.loads,(source/'inputs.jsonl').read_text().splitlines()));rows=list(map(json.loads,(base/'responses.jsonl').read_text().splitlines()));lookup={r['id']:r for r in items}
    assert {r['id'] for r in rows}==set(lookup)
    dest=ROOT/'experiments/.cache'/('jh_r8_policy_component_'+args.name);dest.mkdir();common=dest/'common_judge.jsonl';shutil.copyfile(base/'judge.jsonl',common)
    meta={'scope':'frozen_own_policy_routes_public_tutor_wording_component_not_HTTP','source_run':args.source_run,'dataset_sha256':sha(source/'inputs.jsonl'),'source_responses_sha256':sha(base/'responses.jsonl'),'source_journal_sha256':sha(base/'judge.jsonl'),'evaluator_hashes':evaluator_hashes(),'selection':'own response action/wording using existing predicted_harmful helper, never gold','judge_workers':4,'baseline_tutor':original['tutor']}
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');load_environment();reports=[]
    for variant in ['r8_fixed_harmful','r8_brief_harmful']:
        cls=load('jh_'+variant);pipeline=cls.__new__(cls);changed=[];out=deepcopy(rows)
        async def convert():
            for row in out:
                if row['endpoint']!='tutor' or row.get('error'):continue
                label=predicted_harmful(row['response'])
                if label is None:continue
                state=pipeline._new_state({'messages':lookup[row['id']]['messages']},False);state.label=label
                await pipeline._generate(state);response=state.to_tutor_response()
                assert response['action']==row['response']['action'] and response['citations']==[]
                row['response']=response;changed.append(row['id'])
        asyncio.run(convert())
        (dest/(variant+'_responses.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
        result=score_rows(items,out,4,journal=common);(dest/(variant+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
        report={'variant':variant,'changed':len(changed),'baseline_tutor':original['tutor'],'tutor':result['tutor'],'judge_errors':result['judge_errors'],'invalid':result['invalid'],'changed_ids':changed}
        reports.append(report);print(json.dumps(report),flush=True)
        if result['judge_errors']:break
    (dest/'summary.json').write_text(json.dumps({'manifest':meta,'results':reports},indent=2)+'\n')
    (dest/('JUDGE_FAILED' if reports[-1]['judge_errors'] else 'COMPLETE')).write_text('Policy wording component finished.\n')


if __name__=='__main__':main()
