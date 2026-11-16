"""Offline raw vs protected-normalization char-TFIDF gate; no deployment/API.

Only public400 is fitted. Derived primary/alternative rows are evaluation only.
The common raw similarity groups are transductive split construction only;
every classifier vocabulary is fitted exclusively on its training fold.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings

sys.dont_write_bytecode = True
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
NORMALIZER = ROOT/'experiments/jh_kiwi_normalization_screen.py'
SOURCE_PATHS = {
    'public_tutor': 'public/tutor.jsonl',
    'public_guard': 'public/guard.jsonl',
    'primary388': 'experiments/.cache/jh_transfer_build1/derived_inputs.jsonl',
    'alternative365': 'experiments/.cache/jh_transfer_alternative_build1/derived_inputs.jsonl',
}
TRACK_CATEGORIES = ('normal','emotional_distress','low_signal_input')

def sha(value): return hashlib.sha256(value).hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def last(row): return next(m['content'] for m in reversed(row['messages']) if m['role']=='user')
def dump(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def make_vectorizer():
    return TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)

def fit(texts,labels):
    start,cpu = time.perf_counter(),time.process_time()
    vec=make_vectorizer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=LogisticRegression(C=1000,max_iter=500,class_weight='balanced').fit(vec.fit_transform(texts),labels)
    meta={'vocabulary_size':len(vec.vocabulary_),
          'vocabulary_sha256':sha(json.dumps({k:int(v) for k,v in vec.vocabulary_.items()},sort_keys=True).encode()),
          'train_texts_sha256':sha(json.dumps(texts,ensure_ascii=False).encode()),
          'fit_wall_s':time.perf_counter()-start,'fit_cpu_s':time.process_time()-cpu,
          'iterations':model.n_iter_.tolist(),'warnings':[str(w.message) for w in caught]}
    return vec,model,meta

def infer(vec,model,texts):
    start,cpu=time.perf_counter(),time.process_time()
    prob=model.predict_proba(vec.transform(texts))
    return model.classes_[prob.argmax(axis=1)],prob.max(axis=1),{
        'wall_s':time.perf_counter()-start,'cpu_s':time.process_time()-cpu}

def records(rows,pred,conf,public_by_id,scope,mode,folds=None):
    out=[]
    for i,(row,label,confidence) in enumerate(zip(rows,pred,conf)):
        source_id=row.get('source_id',row['id'])
        original=public_by_id[source_id]
        gold=row['labels']['category']
        assert gold==original['labels']['category']
        out.append({'scope':scope,'mode':mode,'id':row['id'],'source_id':source_id,
                    'original_endpoint':original['_source_endpoint'],'gold':gold,'prediction':str(label),
                    'confidence':float(confidence),'correct':bool(label==gold),
                    'gate_applies':bool(confidence>=.95 and label!='harmful_content'),
                    'fold':int(folds[i]) if folds is not None else None})
    return out

def basic_metric(rows):
    selected=[r for r in rows if r['gate_applies']]
    return {'count':len(rows),'correct':sum(r['correct'] for r in rows),
            'accuracy':sum(r['correct'] for r in rows)/len(rows) if rows else None,
            'high_confidence_nonharmful_count':len(selected),
            'high_confidence_wrong':sum(not r['correct'] for r in selected),
            'high_confidence_accuracy':sum(r['correct'] for r in selected)/len(selected) if selected else None}

def metrics(rows):
    return {**basic_metric(rows),
            'tracked_categories':{category:basic_metric([r for r in rows if r['gold']==category]) for category in TRACK_CATEGORIES},
            'by_original_endpoint':{endpoint:basic_metric([r for r in rows if r['original_endpoint']==endpoint]) for endpoint in ('tutor','guard')}}

def run(output):
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'experiments/.cache').resolve()): raise ValueError('Cache output only')
    if output.exists(): raise FileExistsError('Never overwrite diagnostic')
    tracked=git('diff','--binary','HEAD')
    started,started_cpu=time.perf_counter(),time.process_time()
    spec=importlib.util.spec_from_file_location('protected_normalizer',NORMALIZER)
    normalizer=importlib.util.module_from_spec(spec); spec.loader.exec_module(normalizer)
    fixtures=normalizer.check_normalization()
    def normalized(text):
        result=normalizer.normalize_view(text,True)
        return result if result.strip() else text
    assert normalized('ㅋㅋㅋ')=='ㅋㅋㅋ' and normalized('ㅠㅠ')=='ㅠㅠ'
    assert normalized('힘들어요ㅠㅠ')=='힘들어요 '
    assert normalized('```python\nx = "ㅋㅋ"\nif x != 0:\n    print(x)\n```')=='```python\nx = "ㅋㅋ"\nif x != 0:\n    print(x)\n```'
    sources={}; datasets={}
    for name,path in SOURCE_PATHS.items():
        file=ROOT/path
        datasets[name]=[json.loads(s) for s in file.read_text().splitlines()]
        sources[name]={'path':path,'sha256':sha(file.read_bytes()),'count':len(datasets[name])}
        derivation=file.parent/'derivation.json'
        if derivation.exists():
            sources[name]['derivation']=json.loads(derivation.read_text())
            if sources[name]['derivation']['dataset_sha256'] != sources[name]['sha256']:
                raise ValueError('Derived dataset hash differs: '+name)
    public=[]
    for endpoint in ('tutor','guard'):
        public.extend([{**r,'_source_endpoint':endpoint} for r in datasets['public_'+endpoint]])
    assert len(public)==400
    public_by_id={r['id']:r for r in public}; assert len(public_by_id)==400
    raw=[last(r) for r in public]; transformed=[normalized(s) for s in raw]
    gold=np.array([r['labels']['category'] for r in public])
    # Same raw group recipe and order as experiments/jh_gate_cv.py.
    near=TfidfVectorizer(analyzer='char',ngram_range=(3,5)).fit_transform(raw)
    similarity=(near@near.T).toarray(); group=list(range(len(raw)))
    def find(i):
        while group[i]!=i:
            group[i]=group[group[i]]; i=group[i]
        return i
    for i in range(len(raw)):
        for j in range(i):
            if similarity[i,j]>=.82: group[find(i)]=find(j)
    groups=np.array([find(i) for i in range(len(raw))])
    assert len(set(groups))==375
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=20261116).split(raw,gold,groups))
    fold_ids=np.zeros(len(raw),dtype=int)
    fold_manifest=[]
    for fold,(train,test) in enumerate(folds):
        assert not set(groups[train])&set(groups[test])
        fold_ids[test]=fold
        fold_manifest.append({'fold':fold,'train_ids':[public[i]['id'] for i in train],
                              'test_ids':[public[i]['id'] for i in test]})
    allsets={'public400':public,'primary388':datasets['primary388'],'alternative365':datasets['alternative365']}
    assert len(allsets['primary388'])==388 and len(allsets['alternative365'])==365
    normalization_changes=[]; transformation_counts={}
    for name,rows in allsets.items():
        changed=[]; fallbacks=[]
        for row in rows:
            source=last(row); target=normalized(source)
            if source!=target:
                changed.append(row['id'])
                normalization_changes.append({'dataset':name,'id':row['id'],'gold':row['labels']['category'],
                                              'original':source,'normalized':target})
            if not normalizer.normalize_view(source,True).strip(): fallbacks.append(row['id'])
        transformation_counts[name]={'count':len(rows),'changed_count':len(changed),'changed_ids':changed,
                                     'empty_result_raw_fallback_count':len(fallbacks),'fallback_ids':fallbacks}
    results={}; predictions=[]; fits={}
    for mode,texts in [('raw',raw),('normalized',transformed)]:
        pred=np.empty(len(public),dtype=object); conf=np.zeros(len(public)); fit_meta=[]
        for fold,(train,test) in enumerate(folds):
            vec,model,meta=fit([texts[i] for i in train],gold[train])
            pp,cc,infer_meta=infer(vec,model,[texts[i] for i in test]); pred[test]=pp;conf[test]=cc
            fit_meta.append({'fold':fold,**meta,'inference':infer_meta})
        result_rows=records(public,pred,conf,public_by_id,'public_oof',mode,fold_ids)
        predictions.extend(result_rows); results[mode]={'public_oof':metrics(result_rows)}
        vec,model,meta=fit(texts,gold); fits[mode]={'folds':fit_meta,'full_public':meta}
        for scope,rows in [('public_in_sample',public),('primary388',allsets['primary388']),('alternative365',allsets['alternative365'])]:
            evaluation=[last(r) if mode=='raw' else normalized(last(r)) for r in rows]
            pp,cc,infer_meta=infer(vec,model,evaluation)
            result_rows=records(rows,pp,cc,public_by_id,scope,mode)
            predictions.extend(result_rows); results[mode][scope]={**metrics(result_rows),'inference':infer_meta}
        print(mode,{s:(v['correct'],v['count'],v['high_confidence_nonharmful_count'],v['high_confidence_wrong']) for s,v in results[mode].items()},flush=True)
    raw_records={(r['scope'],r['id']):r for r in predictions if r['mode']=='raw'}
    changes=[]
    for row in predictions:
        if row['mode']!='normalized':continue
        old=raw_records[row['scope'],row['id']]
        if old['prediction']!=row['prediction'] or old['gate_applies']!=row['gate_applies']:
            changes.append({'scope':row['scope'],'id':row['id'],'gold':row['gold'],'raw':old,'normalized':row})
    category_changes={}
    for scope in results['raw']:
        category_changes[scope]={}
        for category in TRACK_CATEGORIES:
            relevant=[r for r in predictions if r['mode']=='normalized' and r['scope']==scope and r['gold']==category]
            category_changes[scope][category]={
                'fixed_errors':sum(r['correct'] and not raw_records[scope,r['id']]['correct'] for r in relevant),
                'introduced_errors':sum(not r['correct'] and raw_records[scope,r['id']]['correct'] for r in relevant),
                'removed_high_confidence_errors':sum(raw_records[scope,r['id']]['gate_applies'] and not raw_records[scope,r['id']]['correct'] and not(r['gate_applies'] and not r['correct']) for r in relevant),
                'added_high_confidence_errors':sum(r['gate_applies'] and not r['correct'] and not(raw_records[scope,r['id']]['gate_applies'] and not raw_records[scope,r['id']]['correct']) for r in relevant)}
    assert git('diff','--binary','HEAD')==tracked,'Tracked worktree changed during screen'
    manifest={'scope':'offline_public_only_gate_normalization_screen_not_holdout','sources':sources,
              'git_head':git('rev-parse','HEAD').strip(),'script_sha256':sha(Path(__file__).read_bytes()),
              'normalizer_dependency':str(NORMALIZER),'normalizer_sha256':sha(NORMALIZER.read_bytes()),
              'versions':{p:importlib.metadata.version(p) for p in ['scikit-learn','numpy','scipy','threadpoolctl']},
              'python':sys.version,'grouping':{'source':'raw last-user text','threshold':.82,'groups':375,'folds':5,'seed':20261116},
              'classifier':{'C':1000,'threshold':.95,'exclude_predicted_category':'harmful_content','max_iter':500,'class_weight':'balanced',
                            'analyzer':'char','ngram_range':[2,5],'min_df':2,'max_features':60000,'sublinear_tf':True},
              'transformation_counts':transformation_counts,'fits':fits,
              'total_wall_s':time.perf_counter()-started,'total_cpu_s':time.process_time()-started_cpu,
              'limitations':['Not production behavior; no LLM fallback is replayed and no endpoint is modified.',
                             'High confidence is uncalibrated and means >=.95 plus predicted category != harmful_content.',
                             'Public in-sample metrics are fitting diagnostics, not unseen performance.',
                             'Derived labels are inherited and noisy; derived data are never fitted; source-public overlap is intentional transfer, not independent holdout.',
                             'Raw full-public TFIDF defines groups only. Classifier vocabulary/IDF/model fit exclusively trainfold.',
                             'Normalization matches prior protected limited NFKC/space/jamo transform, except empty output falls back to raw input.',
                             'No parameter grid, augmentation, embedding or API calls.']}
    summary=['# Offline normalized gate screen','','C=1000, threshold=.95, exclude predicted harmful_content.','','| Scope | Raw correct / N | Normalized correct / N | Raw applies / wrong | Normalized applies / wrong |','|---|---:|---:|---:|---:|']
    for scope in results['raw']:
        a,b=results['raw'][scope],results['normalized'][scope]
        summary.append(f"| {scope} | {a['correct']} / {a['count']} | {b['correct']} / {b['count']} | {a['high_confidence_nonharmful_count']} / {a['high_confidence_wrong']} | {b['high_confidence_nonharmful_count']} / {b['high_confidence_wrong']} |")
    output.mkdir(exist_ok=False)
    for name,value in [('manifest',manifest),('result',results),('folds',fold_manifest),('category_changes',category_changes),('validation',{'protected_fixtures':fixtures,'empty_raw_fallback_passed':True})]:dump(output/(name+'.json'),value)
    for name,rows in [('predictions',predictions),('prediction_changes',changes),('normalization_changes',normalization_changes)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (output/'summary.md').write_text('\n'.join(summary)+'\n')
    (output/'script.py').write_bytes(Path(__file__).read_bytes())
    print('\n'.join(summary));print('OUTPUT',output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'experiments/.cache/jh_norm_gate_r3_screen1'))
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output)
