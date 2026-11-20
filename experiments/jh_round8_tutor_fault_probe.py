"""Tutor behavior under an immediate API failure; no judge or live score."""
import argparse,asyncio,hashlib,json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from core.kb import load_chunks
from core.search import Searcher
from pipelines import load
from experiments.jh_tutor_mlp_compare import SOURCE,build_items,load_public
from experiments.jh_round8_fault_probe import FailedLLM


async def collect(pipeline,items):
    sem=asyncio.Semaphore(64)
    async def one(item):
        async with sem:
            state=await pipeline.tutor({'messages':item['messages']})
            return {'id':item['id'],'response':state.to_tutor_response(),
                    'predicted_category':state.category.value,
                    'fallback':bool(state.debug.get('r8_tutor_failure_input')),'errors':state.errors}
    return await asyncio.gather(*(one(item) for item in items))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);ap.add_argument('--alternative',action='store_true');args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_tutor_fault_'+args.name);dest.mkdir()
    originals={r['id']:r for r in load_public()};derived,_=build_items(SOURCE,args.alternative)
    items=[{**originals[r['source_id']],'id':r['id'],'messages':r['messages']} for r in derived if originals[r['source_id']]['endpoint']=='tutor']
    reports=[]
    for name in ['r7_binary_guard','r8_tutor_failure']:
        p=load('jh_'+name)(Searcher(load_chunks(ROOT/'kb')),FailedLLM());rows=asyncio.run(collect(p,items))
        (dest/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        errors=Counter(e for r in rows for e in r['errors'])
        report={'candidate':name,'count':len(rows),
                'action_correct':sum(r['response']['action']==i['labels']['action'] for i,r in zip(items,rows)),
                'category_correct':sum(r['predicted_category']==i['labels']['category'] for i,r in zip(items,rows)),
                'fallback_count':sum(r['fallback'] for r in rows),'errors':dict(errors)}
        reports.append(report);print(json.dumps(report),flush=True)
    out={'scope':'immediate_failure_public_tutor_behavior_not_judged_score_not_network_latency',
         'api_calls':0,'alternative':args.alternative,
         'dataset_sha256':hashlib.sha256(json.dumps(items,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':reports}
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');(dest/'COMPLETE').write_text('Offline tutor fault diagnostic completed.\n')


if __name__=='__main__':main()
