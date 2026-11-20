"""Freeze a public manual A run, vary only its numeric explanation stage.

Not HTTP/end-to-end latency: remaining budget is reconstructed from recorded A
wall time. Identical judge payloads reuse the first available recorded result.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from experiments.jh_manual_tutor_compare import load_proposal
from experiments.jh_compare import score_rows,evaluator_hashes
from experiments import jh_guard_fusion_compare as runner
from experiments.jh_environment import load_environment


def remaining_budget(seconds):
    if not isinstance(seconds,(float,int)) or not 0<=seconds<=35:raise ValueError('Invalid recorded timing')
    return max(0.,26.-seconds)


async def explain(pipeline,items,baseline,path):
    from core.schema import Label
    sem=asyncio.Semaphore(64);diagnostics=[]
    async def one(item):
        async with sem:
            row=deepcopy(baseline[item['id']]);r=row['response'];started=time.monotonic()
            if r['action']=='answer' and len(r['citations'])==1:
                state=pipeline._new_state({'messages':item['messages']},False)
                state.label=Label.NORMAL;state.section=r['citations'][0]
                state.answer=r['answer'];state.citations=list(r['citations'])
                state._source_extract_deadline=started+remaining_budget(row['seconds'])
                await pipeline._explain(state)
                row['response']={'action':state.action.value,'answer':state.answer,'citations':state.citations}
                diagnostics.append({'id':item['id'],'errors':list(state.errors),
                    'attempted':state.debug.get('r7_apply_attempted') is True,
                    'completed':state.debug.get('r7_apply_completed') is True,
                    'tax_scope':state.debug.get('r7_explicit_tax_scope') is True,
                    'changed':row['response']!=r,'added_seconds':time.monotonic()-started})
            row['seconds']=time.monotonic()-started
            with (path/'responses.jsonl').open('a') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
            return row
    started=time.monotonic();rows=await asyncio.gather(*(one(item) for item in items))
    report={'component_batch_seconds':time.monotonic()-started,'attempted':sum(x['attempted'] for x in diagnostics),
        'completed':sum(x['completed'] for x in diagnostics),'tax_scope':sum(x['tax_scope'] for x in diagnostics),
        'changed':sum(x['changed'] for x in diagnostics),'stage_errors':dict(Counter(e for x in diagnostics for e in x['errors']))}
    runner.write_json(path/'diagnostics.json',diagnostics);runner.write_json(path/'runtime.json',report)
    return rows,report


def main():
    from core.kb import load_chunks
    from core.search import Searcher
    from core.llm import LLM
    from pipelines import load
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True)
    ap.add_argument('--source-run',required=True);ap.add_argument('--order',nargs='+',required=True,choices=['r8_value_prompt','r8_value_anchored','r7_tax_concrete','r7_tax_budget12','r7_apply_scoped']);args=ap.parse_args()
    if any(not s.replace('_','').isalnum() for s in [args.name,args.source_run]) or len(set(args.order))!=len(args.order):ap.error('Unique candidates/simple names required')
    source=ROOT/'experiments/.cache'/('jh_manual_tutor_'+args.source_run)
    if not (source/'COMPLETE').exists():raise ValueError('Only completed manual public runs')
    plan=json.loads((source/'plan.json').read_text());items,provenance=load_proposal(ROOT/plan['proposal_path'])
    if provenance['proposal_sha256']!=plan['proposal_sha256']:raise ValueError('Proposal changed')
    raw=(source/'inputs.jsonl').read_bytes()
    if runner.digest(raw)!=plan['dataset_sha256'] or list(map(json.loads,raw.splitlines()))!=items:raise ValueError('Input changed')
    old=source/'cm_char_mlp50';manifest=json.loads((old/'manifest.json').read_text())
    if manifest['variant']!='jh_cm_char_mlp50' or manifest['evaluator_hashes']!=evaluator_hashes():raise ValueError('Baseline identity mismatch')
    journal_rows=list(map(json.loads,(old/'judge.jsonl').read_text().splitlines()))
    if len({r['key'] for r in journal_rows})!=len(journal_rows):raise ValueError('Ambiguous baseline journal')
    original=json.loads((old/'result.json').read_text())
    if original['invalid'] or original['judge_errors']:raise ValueError('Incomplete baseline scoring')
    baseline={r['id']:r for r in map(json.loads,(old/'responses.jsonl').read_text().splitlines())}
    if set(baseline)!={r['id'] for r in items}:raise ValueError('Incomplete response IDs')
    dest=ROOT/'experiments/.cache'/('jh_r7_tutor_component_'+args.name);dest.mkdir()
    meta={**provenance,'scope':'frozen_A_numeric_component_not_http_not_private_not_holdout',
        'source_run':args.source_run,'dataset_sha256':plan['dataset_sha256'],
        'source_responses_sha256':runner.digest((old/'responses.jsonl').read_bytes()),
        'source_journal_sha256':runner.digest((old/'judge.jsonl').read_bytes()),
        'script_sha256':runner.digest(Path(__file__).read_bytes()),'order':args.order,
        'protected':runner.protected_hashes(ROOT),'pipelines':runner.pipeline_hashes(ROOT),
        'evaluator_hashes':evaluator_hashes(),'remaining_budget':'26 minus recorded A total wall time; stage simulation only',
        'judge_reuse':'identical payload first recorded judgment, common append-only journal', 'workers':64}
    runner.write_json(dest/'manifest.json',meta);load_environment();searcher=Searcher(load_chunks(ROOT/'kb'))
    collected={}
    for name in args.order:
        path=dest/name;path.mkdir();pipeline=load('jh_'+name)(searcher,LLM())
        async def run():
            try:return await explain(pipeline,items,baseline,path)
            finally:
                await pipeline.llm._chat.close()
                if pipeline.llm._embed is not None:await pipeline.llm._embed.close()
        collected[name]=asyncio.run(run());print('COLLECTED '+name,flush=True)
    common=dest/'common_judge.jsonl';shutil.copyfile(old/'judge.jsonl',common)
    reports=[]
    for name in args.order:
        rows,runtime=collected[name];result=score_rows(items,rows,32,journal=common)
        runner.write_json(dest/name/'result.json',result)
        r={'candidate':name,'baseline_tutor':original['tutor'],'tutor':result['tutor'],
            'invalid':result['invalid'],'judge_errors':result['judge_errors'],'runtime':runtime};reports.append(r);print(json.dumps(r),flush=True)
    if runner.pipeline_hashes(ROOT)!=meta['pipelines'] or runner.protected_hashes(ROOT)!=meta['protected']:raise ValueError('Code changed during component run')
    runner.write_json(dest/'summary.json',{'manifest':meta,'results':reports});(dest/'COMPLETE').write_text('Numeric explanation component complete; not end-to-end.\n')


if __name__=='__main__':main()
