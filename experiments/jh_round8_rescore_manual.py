"""Rejudge frozen public manual responses after an API outage, without rerunning servers."""
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.jh_environment import load_environment
from experiments.jh_manual_tutor_compare import load_proposal
from experiments.jh_compare import score_rows,evaluator_hashes


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if any(not s.replace('_','').isalnum() for s in (args.source_run,args.name)) or not 1<=args.workers<=8:
        ap.error('Simple names and 1–8 judge workers required')
    source=ROOT/'experiments/.cache'/('jh_manual_tutor_'+args.source_run)
    plan=json.loads((source/'plan.json').read_text())
    items,provenance=load_proposal(ROOT/plan['proposal_path'])
    raw=[json.loads(s) for s in (source/'inputs.jsonl').read_text().splitlines()]
    assert items==raw and sha(source/'inputs.jsonl')==plan['dataset_sha256']
    assert provenance['proposal_sha256']==plan['proposal_sha256']
    dest=ROOT/'experiments/.cache'/('jh_r8_rejudge_'+args.name);dest.mkdir()
    manifest={'scope':'rejudged_frozen_public_manual_responses_not_new_collection',
              'source_run':args.source_run,'source_commit':plan['commit'],
              'dataset_sha256':plan['dataset_sha256'],'proposal_sha256':plan['proposal_sha256'],
              'judge_workers':args.workers,'evaluator_hashes':evaluator_hashes(),
              'judge_reuse':'common identical public payload within this new pass; original failure results untouched',
              'response_hashes':{},'script_sha256':sha(Path(__file__))}
    for name in plan['order']:
        original=json.loads((source/name/'manifest.json').read_text())
        assert original['evaluator_hashes']==evaluator_hashes()
        manifest['response_hashes'][name]=sha(source/name/'responses.jsonl')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    load_environment();reports=[]
    for name in plan['order']:
        rows=[json.loads(s) for s in (source/name/'responses.jsonl').read_text().splitlines()]
        result=score_rows(items,rows,args.workers,journal=dest/'common_judge.jsonl')
        (dest/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        report={k:result[k] for k in ('tutor','invalid','judge_errors')};report['candidate']=name
        reports.append(report);print(json.dumps(report),flush=True)
        (dest/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
        if result['judge_errors']:
            (dest/'JUDGE_FAILED').write_text('Stop before another candidate; preserve this attempt.\n')
            return
    (dest/'COMPLETE').write_text('Frozen response rejudging complete.\n')


if __name__=='__main__':main()
