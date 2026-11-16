"""Replay our recorded public draft candidates; unchanged evaluator, no new generation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from jh_tutor_generation_probe import ROOT, source_data
from jh_compare import evaluator_hashes, score_rows
from pipelines.kimjaehak.draft_selection import select


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=True)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--method',choices=('consensus','longest'),required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--collect-only',action='store_true')
    ap.add_argument('--score-only',action='store_true')
    args=ap.parse_args()
    source=args.source.resolve()
    if not source.is_relative_to(ROOT/'experiments/.cache') or not args.name.replace('_','').isalnum():
        ap.error('Use local public diagnostic recordings and a safe name')
    meta=json.loads((source/'manifest.json').read_text())
    if meta['scope']!='frozen_own_routing_source_generation_component_not_http' or not meta.get('record_candidate_comparison'):
        ap.error('Source did not record its own public candidates')
    if meta['evaluator_hashes']!=evaluator_hashes():raise ValueError('Evaluator changed')
    _,items,_,_=source_data(ROOT/meta['source'])
    rows=list(map(json.loads,(source/'responses.jsonl').read_text().splitlines()))
    if len(rows)!=len(items) or {r['id'] for r in rows}!={i['id'] for i in items}:
        raise ValueError('Incomplete source')
    old_result=json.loads((source/'result.json').read_text())
    target=ROOT/'experiments/.cache'/('jh_draft_'+args.name)
    provenance={'scope':'public_frozen_draft_selection_only','method':args.method,'source':str(source.relative_to(ROOT)),
                'source_response_sha256':hashlib.sha256((source/'responses.jsonl').read_bytes()).hexdigest(),
                'count':len(items),'evaluator_hashes':evaluator_hashes(),
                'source_selected_tutor':old_result['revised_tutor'],'source_first_valid_tutor':old_result['first_valid_tutor']}
    if args.score_only:
        existing=json.loads((target/'manifest.json').read_text())
        if any(existing[k]!=v for k,v in provenance.items()):raise ValueError('Provenance differs')
        report=existing
        revised=list(map(json.loads,(target/'responses.jsonl').read_text().splitlines()))
    else:
        target.mkdir(exist_ok=False)
        revised=copy.deepcopy(rows)
        compared=changed=0
        for row in revised:
            if row.get('generation_candidates'):
                answers=row['generation_candidates']
                if not isinstance(answers,list) or not all(isinstance(a,str) for a in answers):
                    raise ValueError('Invalid recorded candidates')
                i=select(answers,args.method)
                compared+=int(len(answers)>1)
                changed+=int(row['response']['answer']!=answers[i])
                row['response']['answer']=answers[i]
                row['replay_selected']=i
        report={**provenance,'compared':compared,'changed_from_selected':changed,
                'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
        (target/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (target/'responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in revised))
    if args.collect_only:
        print(json.dumps(report),flush=True)
        return
    if (target/'result.json').exists():raise ValueError('Already scored')
    if not (target/'judge.jsonl').exists():shutil.copyfile(source/'judge.jsonl',target/'judge.jsonl')
    result=score_rows(items,revised,32,journal=target/'judge.jsonl')
    report.update(tutor=result['tutor'],invalid=result['invalid'],judge_errors=result['judge_errors'],result=result)
    (target/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('method','count','source_first_valid_tutor','tutor','changed_from_selected','invalid','judge_errors')}),flush=True)


if __name__=='__main__':main()
