"""Paired input-only diagnostic with one frozen common LLM fallback.

Only pinned public-derived development sets are accepted. Organizer embeddings
are cached locally for this analysis, never imported into deployment or training.
Output scoring and real latency are deliberately excluded from this replay.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from core.schema import Label, POLICY, parse_label
from experiments.jh_environment import load_environment
from experiments.jh_classifier_matrix import metrics
from experiments.jh_guard_fusion_datasets import load_dataset
from pipelines.kimjaehak.calibrated import low_signal
from pipelines.kimjaehak.classifier_models import CategoryModel, public_data
from pipelines.kimjaehak.embedding_identity import same_embedding_model
from pipelines.kimjaehak.learned_harmful import train_harmful_classifier


def gate_label(text, classes, probabilities, threshold, subtype=None):
    if low_signal(text):return Label.LOW_SIGNAL_INPUT
    p=np.asarray(probabilities)
    if p.shape!=(len(classes),) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError('Invalid probabilities')
    i=int(p.argmax());category=str(classes[i])
    if p[i]<threshold:return None
    if category=='harmful_content':
        if subtype is None or subtype[1]<.95:return None
        return {'refuse':Label.HARMFUL_REQUEST,'redirect':Label.HARMFUL_SPEECH}.get(subtype[0])
    return parse_label(category)


async def collect_vectors(texts, expected_model):
    from core.llm import LLM
    load_environment();llm=LLM()
    try:
        if not llm.can_embed or not same_embedding_model(llm.embedding_model,expected_model):
            raise ValueError('Embedding identity mismatch')
        vectors=[];calls=0
        for start in range(0,len(texts),16):
            batch=texts[start:start+16]
            result=await llm.embed(batch,timeout=15)
            CategoryModel._vectors(result,len(batch))
            vectors.extend(result);calls+=1
        return np.asarray(vectors),calls
    finally:
        await llm._chat.close()
        if llm._embed is not None:await llm._embed.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name',required=True);ap.add_argument('--run',required=True)
    ap.add_argument('--dataset',required=True,choices=['paraphrase','alternative','intent'])
    ap.add_argument('--reuse-vectors')
    args=ap.parse_args()
    for name in (args.name,args.run,args.reuse_vectors):
        if name and not name.replace('_','').replace('-','').isalnum():ap.error('Simple cache names required')
    root=ROOT/'experiments/.cache'/('jh_guard_fusion_'+args.run)
    plan=json.loads((root/'plan.json').read_text());raw=(root/'inputs.jsonl').read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=plan['dataset_sha256']:raise ValueError('Input checksum mismatch')
    items=[json.loads(s) for s in raw.splitlines()]
    expected,provenance=load_dataset(args.dataset,ROOT)
    if items!=expected:raise ValueError('Only pinned public development inputs accepted')
    fallback_path=root/'r7_llm_input';config=json.loads((fallback_path/'manifest.json').read_text())
    if config['variant']!='jh_r7_llm_input':raise ValueError('Wrong common fallback')
    if not (fallback_path/'result.json').exists():raise ValueError('Incomplete fallback run')
    fallback_rows=[json.loads(s) for s in (fallback_path/'responses.jsonl').read_text().splitlines()]
    fallback={r['id']:r['response']['input'] for r in fallback_rows}
    if len(fallback)!=len(items) or set(fallback)!={r['id'] for r in items}:raise ValueError('Incomplete IDs')
    texts=[next(m['content'] for m in reversed(r['messages']) if m['role']=='user') for r in items]
    dest=ROOT/'experiments/.cache'/('jh_r7_replay_'+args.name);dest.mkdir()
    train,train_texts,train_vectors,embedding_model=public_data()
    if args.reuse_vectors:
        previous=ROOT/'experiments/.cache'/('jh_r7_replay_'+args.reuse_vectors)
        manifest=json.loads((previous/'manifest.json').read_text())
        if manifest['dataset_sha256']!=digest or manifest['embedding_model']!=embedding_model:
            raise ValueError('Cached vector provenance mismatch')
        vectors=np.load(previous/'vectors.npz')['vectors'];calls=0
    else:
        vectors,calls=asyncio.run(collect_vectors(texts,embedding_model))
    CategoryModel._vectors(vectors,len(texts))
    np.savez_compressed(dest/'vectors.npz',vectors=vectors)
    manifest={'scope':'frozen_fallback_input_only_counterfactual_not_live_score','count':len(items),
        'dataset_sha256':digest,'dataset':args.dataset,'embedding_model':embedding_model,
        'embedding_calls':calls,'fallback_run':args.run,'fallback_commit':plan['current_commit'],
        'fallback_responses_sha256':hashlib.sha256((fallback_path/'responses.jsonl').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'not_independent_holdout':True,'latency_and_output_effects_not_modeled':True}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from pipelines import load
    load('jh_cm_char_mlp50')
    from pipelines.kimjaehak.round7_hybrid import BlendModel,ConcatModel
    from pipelines.kimjaehak.round7_guard import BoundaryModel
    labels=[r['labels']['category'] for r in train]
    fitted={
        'char':CategoryModel('char_mlp').fit(train_texts,labels),
        'boundary':BoundaryModel('char_mlp').fit(train_texts,labels),
        'concat':ConcatModel('dense_mlp').fit(train_texts,labels,train_vectors),
        'blend':BlendModel().fit(train_texts,labels,train_vectors)}
    probs={name:m.probabilities(texts,vectors) for name,m in fitted.items()}
    hv,hm=train_harmful_classifier(1000);hp=hm.predict_proba(hv.transform(texts))
    subtypes=[(str(hm.classes_[p.argmax()]),float(p.max())) for p in hp]
    thresholds={'char':.8556677733093049,'boundary':.8720184128780901,
        'concat':.9011610502863048,'blend':.8208260969823079,'fastblend':.8208260969823079}
    gold=np.array([r['labels']['category'] for r in items]);predictions={};reports=[]
    for name in ['llm_only',*thresholds]:
        rows=[];direct=[]
        for i,item in enumerate(items):
            label=None;model_name='blend' if name=='fastblend' else name
            if name=='fastblend':
                cp=probs['char'][i];classes=fitted['char'].classes
                if str(classes[cp.argmax()])!='harmful_content':
                    label=gate_label(texts[i],classes,cp,.9641535987031538,subtypes[i])
            if label is None and name!='llm_only':
                label=gate_label(texts[i],fitted[model_name].classes,probs[model_name][i],thresholds[name],subtypes[i])
            prediction=fallback[item['id']] if label is None else {
                'category':POLICY[label].category.value,'decision':POLICY[label].decision.value}
            rows.append({'id':item['id'],'input':prediction,'direct':label is not None})
            direct.append(label is not None)
        pred=np.array([r['input']['category'] for r in rows]);direct=np.array(direct)
        predictions[name]=pred
        report={'name':name,'metrics':metrics(gold,pred),'direct':int(direct.sum()),
            'direct_errors':int((pred[direct]!=gold[direct]).sum()),'threshold':thresholds.get(name)}
        if name not in ('char','llm_only'):
            baseline=predictions['char']
            report.update(corrected_vs_char=int(((pred==gold)&(baseline!=gold)).sum()),
                regressed_vs_char=int(((pred!=gold)&(baseline==gold)).sum()))
        reports.append(report)
        (dest/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    (dest/'summary.json').write_text(json.dumps({'manifest':manifest,'results':reports},indent=2)+'\n')
    (dest/'COMPLETE').write_text('Fixed fallback input replay completed. Not a live guard score.\n')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('REPLAY_FAILED '+type(exc).__name__,file=sys.stderr)
        raise SystemExit(1)
