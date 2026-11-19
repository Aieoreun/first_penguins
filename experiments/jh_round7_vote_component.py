"""One set of public-only grounding votes, several descriptive aggregation replays.

This is not a submitted server or an end-to-end score. Recorded A predictions
select ALLOW inputs; gold labels are used only after generation for reporting.
The staged variants cannot inherit these parallel timings or deadline outcomes.
"""
import argparse
import asyncio
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def selected_votes(candidate,votes,mode):
    from pipelines.kimjaehak.voted_output import covered_positions
    finished=[v for v in votes if v['completed']]
    if mode in {'first2','first3'}:
        count=int(mode[-1]);first=votes[:count]
        if all(v['completed'] and not v['errors'] for v in first):
            masks=[covered_positions(candidate,v['unsupported']) for v in first]
            if all(m==masks[0] for m in masks[1:]):return first,True
    if mode=='early3':
        ordered=sorted(finished,key=lambda v:(v['seconds'],v['index']))
        for end in range(3,len(ordered)+1):
            prefix=ordered[:end]
            if any(v['errors'] for v in prefix):continue
            masks=[covered_positions(candidate,v['unsupported']) for v in prefix]
            if all(m==masks[0] for m in masks[1:]):
                return sorted(prefix,key=lambda v:v['index']),end<len(votes)
    elif mode not in {'all5','first2','first3'}:raise ValueError('Unknown aggregation')
    return finished,False


def aggregate(pipeline,candidate,votes,mode):
    from pipelines.kimjaehak.voted_output import consensus_index
    from pipelines.kimjaehak.span_tagger import refine_spans
    chosen,early=selected_votes(candidate,votes,mode);good=[v for v in chosen if not v['errors']]
    spans=good[consensus_index(candidate,[v['unsupported'] for v in good])]['unsupported'] if good else []
    if spans:
        predicted,confidence=pipeline.span_tagger.predict(candidate)
        spans,_=refine_spans(candidate,spans,predicted,confidence,pipeline.learned_span_threshold)
    return spans,{'selected_good_votes':len(good),'early':early}


async def collect(pipeline,items,dest):
    from pipelines.kimjaehak.context_output import Pipeline as ContextOutput
    from pipelines.kimjaehak.peace_guard_fusion import isolated_state
    sem=asyncio.Semaphore(64)
    async def one(item):
        async with sem:
            state=pipeline._new_state({k:item[k] for k in ('messages','candidate_answer')},True)
            await pipeline._search(state,extra=state.candidate_answer or '')
            if not state.chunks:raise ValueError('No positive lexical source')
            state.section=state.chunks[0].section_id
            # Embeddings append after positive BM25. One-section grounding thus
            # has the same payload, without making query-embedding API calls.
            query='\n'.join(m['content'] for m in state.messages)+'\n'+state.candidate_answer
            if max(pipeline.searcher.bm25.scores(query))<=0:raise ValueError('Dense-only source not supported')
            started=time.monotonic();votes=[{'index':i,'completed':False,'seconds':None,'errors':[],'unsupported':[]} for i in range(5)]
            async def vote(i):
                trial=isolated_state(state)
                try:await ContextOutput._check_output(pipeline,trial)
                except asyncio.CancelledError:raise
                except Exception as exc:trial.errors.append('vote:'+type(exc).__name__)
                votes[i].update(completed=True,seconds=time.monotonic()-started,
                    errors=list(trial.errors),unsupported=list(trial.unsupported))
            tasks=[asyncio.create_task(vote(i)) for i in range(5)]
            try:await asyncio.wait_for(asyncio.gather(*tasks),pipeline.guard_ground_timeout)
            except asyncio.TimeoutError:pass
            finally:
                for task in tasks:
                    if not task.done():task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
            row={'id':item['id'],'source_section':state.section,'votes':votes}
            with (dest/'votes.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            return row
    return await asyncio.gather(*(one(item) for item in items))


def main():
    from experiments.jh_guard_fusion_datasets import load_dataset
    from experiments.jh_compare import evaluator_hashes
    from experiments.jh_environment import load_environment
    from experiments.jh_guard_fusion_compare import protected_hashes,pipeline_hashes
    from experiments.jh_neural_peace_report import grounding_metrics
    from core.kb import load_chunks
    from core.search import Searcher
    from core.llm import LLM
    from pipelines import load
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True);ap.add_argument('--source-run',required=True)
    ap.add_argument('--dataset',choices=['novel','paraphrase'],required=True);args=ap.parse_args()
    if any(not s.replace('_','').isalnum() for s in [args.name,args.source_run]):ap.error('Simple names required')
    source=ROOT/'experiments/.cache'/('jh_guard_fusion_'+args.source_run)
    items,provenance=load_dataset(args.dataset,ROOT)
    raw=(source/'inputs.jsonl').read_bytes();plan=json.loads((source/'plan.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=plan['dataset_sha256'] or [json.loads(s) for s in raw.splitlines()]!=items:
        raise ValueError('Only pinned public development datasets')
    baseline=source/'cm_char_mlp50';manifest=json.loads((baseline/'manifest.json').read_text())
    if manifest['variant']!='jh_cm_char_mlp50' or manifest['evaluator_hashes']!=evaluator_hashes():
        raise ValueError('Wrong baseline or evaluator changed')
    result=json.loads((baseline/'result.json').read_text())
    if result['invalid'] or result['judge_errors']:raise ValueError('Incomplete baseline')
    prior={r['id']:r for r in map(json.loads,(baseline/'responses.jsonl').read_text().splitlines())}
    if set(prior)!={r['id'] for r in items}:raise ValueError('Incomplete response IDs')
    selected=[r for r in items if r.get('candidate_answer') is not None
        and prior[r['id']]['response']['input']['decision']=='allow']
    dest=ROOT/'experiments/.cache'/('jh_r7_votes_'+args.name);dest.mkdir()
    load_environment();pipeline=load('jh_cm_char_mlp50')(Searcher(load_chunks(ROOT/'kb')),LLM())
    if pipeline.output_sections!=1:raise ValueError('Only positive lexical top-one source supported')
    metadata={'scope':'public_recorded_five_vote_component_not_end_to_end_not_private',
        'dataset':args.dataset,'dataset_sha256':plan['dataset_sha256'],'selected_count':len(selected),
        'selection':'baseline predicted ALLOW, not gold','source_run':args.source_run,
        'source_responses_sha256':hashlib.sha256((baseline/'responses.jsonl').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluator_hashes':evaluator_hashes(),'protected':protected_hashes(ROOT),
        'pipelines':pipeline_hashes(ROOT),'workers':64,'calls_per_selected_item':5,
        'ground_timeout':pipeline.guard_ground_timeout,
        'limitations':['No input/safety workload; not real request latency or submission score.',
            'Staged subset replay assumes remaining parallel votes available; not staged timeout simulation.',
            'Uses previously developed public-derived questions, not independent evaluation.']}
    (dest/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    async def run():
        try:return await collect(pipeline,selected,dest)
        finally:
            await pipeline.llm._chat.close()
            if pipeline.llm._embed is not None:await pipeline.llm._embed.close()
    votes={r['id']:r for r in asyncio.run(run())};reports=[]
    for mode in ('all5','first2','first3','early3'):
        rows=[];early=0;used=0
        for item in selected:
            spans,stats=aggregate(pipeline,item['candidate_answer'],votes[item['id']]['votes'],mode)
            response=deepcopy(prior[item['id']]['response'])
            response['output'].update(grounded=not spans,unsupported=spans)
            rows.append({'id':item['id'],'endpoint':'guard','response':response})
            early+=stats['early'];used+=stats['selected_good_votes']
        report={'aggregation':mode,'count':len(rows),'early':early,'selected_good_votes':used,
                'grounding':grounding_metrics(selected,rows)}
        reports.append(report);print(json.dumps(report),flush=True)
    if (evaluator_hashes()!=metadata['evaluator_hashes'] or protected_hashes(ROOT)!=metadata['protected']
            or pipeline_hashes(ROOT)!=metadata['pipelines']):raise ValueError('Frozen code or data changed')
    flat=[v for r in votes.values() for v in r['votes']]
    metadata['recorded_votes']={'scheduled':len(flat),'completed':sum(v['completed'] for v in flat),
        'deadline_cancelled':sum(not v['completed'] for v in flat),
        'errors':dict(Counter(e for v in flat for e in v['errors']))}
    (dest/'summary.json').write_text(json.dumps({'manifest':metadata,'results':reports},ensure_ascii=False,indent=2)+'\n')
    (dest/'COMPLETE').write_text('Common-vote public component analysis complete.\n')


if __name__=='__main__':main()
