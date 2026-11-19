"""Sequential full-pipeline comparison on the previously reviewed KB contrasts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.jh_environment import load_environment
from experiments.jh_kb_contrast_probe import build_items
from experiments.jh_compare import score_rows
from experiments import jh_guard_fusion_compare as runner


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True)
    ap.add_argument('--order',nargs='+',choices=sorted(runner.CANDIDATES),required=True)
    args=ap.parse_args()
    if not args.name.replace('_','').isalnum() or len(set(args.order))!=len(args.order):
        ap.error('Simple name and unique candidates required')
    items,provenance=build_items(ROOT/'experiments/.cache/jh_r6_kb_contrast_draft/proposal.json',
        ROOT/'experiments/.cache/jh_r6_kb_contrast_review/review.json')
    frozen=(ROOT/'experiments/.cache/jh_kb_contrast_frozen_v1/derived_inputs.jsonl').read_text()
    if [json.loads(s) for s in frozen.splitlines()]!=items:
        raise ValueError('Reviewed contrasts differ from frozen dataset')
    encoded=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items)
    load_environment()
    plan={**provenance,'scope':'reviewed_kb_contrast16_round7_development_not_private_not_holdout',
        'dataset_sha256':runner.digest(encoded.encode()),'order':args.order,'count':len(items),
        'protected':runner.protected_hashes(ROOT),'pipelines':runner.pipeline_hashes(ROOT),
        'bootstrap_sha256':runner.digest(runner.BOOTSTRAP.encode()),
        'runner_sha256':runner.digest(Path(__file__).read_bytes()),
        'chat_model':os.environ['OPENAI_MODEL'],'embedding_model':os.environ.get('EMBEDDING_MODEL'),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    dest=ROOT/'experiments/.cache'/('jh_r7_contrast_'+args.name);dest.mkdir()
    runner.write_json(dest/'plan.json',plan);(dest/'inputs.jsonl').write_text(encoded)
    env=runner.server_env(os.environ,ROOT,'env')
    for label in args.order:
        runner.collect_candidate(label,ROOT,items,dest/label,env,32,'env',plan)
        print('COLLECTED '+label,flush=True)
    summary=[]
    for label in args.order:
        root=dest/label;rows=[json.loads(s) for s in (root/'responses.jsonl').read_text().splitlines()]
        result=score_rows(items,rows,32,journal=root/'judge.jsonl')
        runner.write_json(root/'result.json',result)
        by_id={r['id']:r['response'] for r in rows}
        counts={}
        for role in ('supported','unsupported'):
            group=[r for r in items if r['contrast_role']==role]
            counts[role]={'count':len(group),'correct_action':sum(
                by_id[r['id']]['action']==r['labels']['action'] for r in group)}
        entry={'candidate':label,**{k:result[k] for k in ('tutor','invalid','judge_errors')},
               'actions':counts,'runtime':json.loads((root/'runtime.json').read_text())}
        summary.append(entry);print(json.dumps(entry),flush=True)
    if runner.protected_hashes(ROOT)!=plan['protected'] or runner.pipeline_hashes(ROOT)!=plan['pipelines']:
        raise ValueError('Runtime code changed during comparison')
    runner.write_json(dest/'summary.json',summary)
    (dest/'COMPLETE').write_text('Reviewed KB contrasts scored. Development only.\n')


if __name__=='__main__':main()
