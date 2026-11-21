"""Public output embedding screen; no private inputs or deployment data writes."""
import argparse,asyncio,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from experiments.jh_classifier_matrix import group_folds
from experiments.jh_guard_fusion_datasets import load_dataset
from pipelines.kimjaehak.classifier_models import CategoryModel


def data():
    train=[r for r in map(json.loads,(ROOT/'public/guard.jsonl').read_text().splitlines()) if r.get('candidate_answer')]
    suites={name:[r for r in load_dataset(name,ROOT)[0] if r.get('candidate_answer')] for name in ['novel','refusal','cross_safety']}
    texts=list(dict.fromkeys(r['candidate_answer'] for r in train+sum(suites.values(),[])))
    return train,suites,texts


async def collect(dest,texts):
    from experiments.jh_environment import load_environment
    from core.llm import LLM
    load_environment();llm=LLM();vectors=[];calls=0
    try:
        for start in range(0,len(texts),16):
            batch=texts[start:start+16];v=await llm.embed(batch,20);calls+=1
            x=np.asarray(v,dtype=float)
            if x.shape!=(len(batch),1536) or not np.isfinite(x).all():raise ValueError('Invalid output embeddings')
            vectors.extend(v);print('embedded',len(vectors),flush=True)
        np.savez_compressed(dest/'vectors.npz',vectors=np.asarray(vectors))
        meta={'model':llm.embedding_model,'count':len(texts),'embedding_calls':calls,'text_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in texts]}
        (dest/'embeddings.json').write_text(json.dumps(meta,indent=2)+'\n')
    finally:
        await llm._chat.close()
        if llm._embed is not None:await llm._embed.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--score-only',action='store_true');args=ap.parse_args()
    if not args.name.replace('_','').isalnum():ap.error('Simple name required')
    dest=ROOT/'experiments/.cache'/('jh_r8_output_dense_'+args.name)
    train,suites,texts=data()
    if not args.score_only:dest.mkdir();asyncio.run(collect(dest,texts))
    meta=json.loads((dest/'embeddings.json').read_text())
    if meta['text_sha256']!=[hashlib.sha256(t.encode()).hexdigest() for t in texts]:raise ValueError('Source text order changed')
    x=np.load(dest/'vectors.npz')['vectors'];lookup={t:v for t,v in zip(texts,x)}
    gold=np.array(['safe' if r['labels']['safe'] else 'unsafe' for r in train]);tx=[r['candidate_answer'] for r in train];vx=np.array([lookup[t] for t in tx]);groups,folds=group_folds(tx,gold)
    reports=[]
    for kind,c in [('char_lr',100),('dense_lr',10),('dense_lr',100),('dense_mlp',100)]:
        oof=np.empty(len(train));fits=[]
        for a,b in folds:
            m=CategoryModel(kind,c=c).fit([tx[i] for i in a],gold[a],vx[a]);oof[b]=m.probabilities([tx[i] for i in b],vx[b])[:,int(np.flatnonzero(m.classes=='unsafe')[0])];fits.append(m.diagnostics)
        cv=[]
        for t in [.5,.7,.9]:
            p=oof>=t;truth=gold=='unsafe';cv.append({'threshold':t,'false_alarm':int((p&~truth).sum()),'miss':int((~p&truth).sum())})
        m=CategoryModel(kind,c=c).fit(tx,gold,vx);index=int(np.flatnonzero(m.classes=='unsafe')[0]);replay=[]
        for name,rows in suites.items():
            source=ROOT/'experiments/.cache'/('jh_guard_fusion_r8_'+{'novel':'novel01','refusal':'refusal01','cross_safety':'safety01'}[name])/'r7_binary_guard/responses.jsonl'
            base={r['id']:r['response']['output']['safe'] for r in map(json.loads,source.read_text().splitlines()) if r['response'].get('output')}
            out=m.probabilities([r['candidate_answer'] for r in rows],np.array([lookup[r['candidate_answer']] for r in rows]))[:,index]
            for t in [.5,.7,.9]:
                changed=[r for r,p in zip(rows,out) if base[r['id']] and p>=t]
                replay.append({'dataset':name,'threshold':t,'corrected':sum(not r['labels']['safe'] for r in changed),'regressed':sum(r['labels']['safe'] for r in changed)})
        report={'model':kind,'C':c,'fits':fits,'grouped_oof':cv,'frozen_B_replay':replay};reports.append(report);print(json.dumps(report),flush=True)
    result={'scope':'public_output86_grouped_embedding_screen_not_HTTP_or_private_score','embedding_metadata':meta,'training_only':'public/guard.jsonl candidate text and safe label','development_only':['novel','refusal','cross_safety'],'results':reports}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');(dest/'COMPLETE').write_text('Output embedding screen complete.\n')


if __name__=='__main__':main()
