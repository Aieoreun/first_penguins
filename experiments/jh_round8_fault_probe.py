"""API-free injected failure experiment; not real submission or HTTP latency."""
import argparse,asyncio,json,time
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from core.kb import load_chunks
from core.search import Searcher
from pipelines import load
from experiments.jh_guard_fusion_datasets import load_dataset
from experiments.jh_compare import score_rows

class FailedLLM:
    can_embed=False
    embedding_model=''
    async def chat(self,*args,**kwargs):raise TimeoutError('Injected offline fault')
    async def embed(self,*args,**kwargs):raise TimeoutError('Injected offline fault')

async def collect(pipeline,items):
    sem=asyncio.Semaphore(64);errors=Counter();rescues=Counter()
    async def one(item):
        async with sem:
            started=time.monotonic();state=await pipeline.guard({'messages':item['messages'],**({'candidate_answer':item['candidate_answer']} if 'candidate_answer' in item else {})})
            errors.update(state.errors)
            for key in ['r8_failure_input','r8_failure_ground']:rescues[key]+=bool(state.debug.get(key))
            return {'id':item['id'],'endpoint':'guard','response':state.to_guard_response(),'seconds':time.monotonic()-started}
    rows=await asyncio.gather(*(one(i) for i in items));return rows,errors,rescues

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--dataset',choices=['paraphrase','novel','alternative'],required=True);args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_fault_'+args.name);dest.mkdir()
    items,provenance=load_dataset(args.dataset,ROOT);results=[]
    for name in ['r7_binary_guard','r8_failure_input','r8_failure_ground','r8_failure_both']:
        p=load('jh_'+name)(Searcher(load_chunks(ROOT/'kb')),FailedLLM());rows,errors,rescues=asyncio.run(collect(p,items));result=score_rows(items,rows,1,no_judge=True)
        (dest/(name+'_responses.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        report={'candidate':name,'guard':result['guard'],'invalid':result['invalid'],'details':result['guard_details'],'errors':dict(errors),'rescues':dict(rescues)};results.append(report);print(json.dumps(report),flush=True)
    (dest/'results.json').write_text(json.dumps({'scope':'injected_immediate_timeout_public_development_not_live_http_not_private','api_calls':0,'dataset':provenance,'results':results},indent=2)+'\n');(dest/'COMPLETE').write_text('Fault simulation complete.\n')
if __name__=='__main__':main()
