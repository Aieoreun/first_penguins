"""Two predeclared CPU hypotheses; fold-safe classifier/threshold fitting.

H1: Last-user char features plus a half-weight 200-character tail channel for
utterances longer than 200 characters. H2: Class-specific thresholds determined
only from nested training-fold OOF predictions. No API or runtime changes.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

sys.dont_write_bytecode=True
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
CACHE=(ROOT/'experiments/.cache').resolve()
SOURCE_PATHS={
    'public_tutor':'public/tutor.jsonl',
    'public_guard':'public/guard.jsonl',
    'primary388':'experiments/.cache/jh_transfer_build1/derived_inputs.jsonl',
    'alternative365':'experiments/.cache/jh_transfer_alternative_build1/derived_inputs.jsonl',
}
EXPECTED_HASHES={
    'public_tutor':'8f8a74a7747281d89b27cb972169f95401b1082ebfa99dce72af55e0f6dc31e0',
    'public_guard':'4a585c174f15dec205a286ec45e8c14ff28b01bbcedde3097072ca52e3b0c426',
    'primary388':'adfc9ba4b5caccbe87789b59d1c7beccb5f1e64be6ef72ca6339d4e0df7213d0',
    'alternative365':'e92d84b39aa79bb457db1106a929642137a3b879c5b496d30439982cb392798b',
}
TRACK_CATEGORIES=('normal','emotional_distress','low_signal_input')

def last(row):return next(m['content'] for m in reversed(row['messages']) if m['role']=='user')

def make_vectorizer():
    return TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)

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

def records(rows,pred,conf,by_id,scope,mode):
    out=[]
    for row,label,confidence in zip(rows,pred,conf):
        source_id=row.get('source_id',row['id']);original=by_id[source_id]
        gold=row['labels']['category']
        if gold!=original['labels']['category']:raise ValueError('Derived category differs from public source')
        out.append({'scope':scope,'mode':mode,'id':row['id'],'source_id':source_id,
                    'original_endpoint':original['_source_endpoint'],'gold':gold,'prediction':str(label),
                    'confidence':float(confidence),'correct':bool(label==gold),
                    'gate_applies':bool(confidence>=.95 and label!='harmful_content'),'fold':None})
    return out

def tail(text):return text[-200:] if len(text)>200 else ''
def sha(data):return hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def fit(texts,labels,with_tail=False):
    start=time.process_time()
    full=make_vectorizer();x=full.fit_transform(texts)
    suffix=None
    if with_tail:
        suffix=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=20000,sublinear_tf=True)
        z=suffix.fit_transform([tail(t) for t in texts]);x=hstack([x,z*.5]).tocsr()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model=LogisticRegression(C=1000,max_iter=500,class_weight='balanced').fit(x,labels)
    return (full,suffix,model),{'train_count':len(texts),'vocabulary':len(full.vocabulary_),
                'tail_vocabulary':len(suffix.vocabulary_) if suffix else 0,
                'fit_cpu_s':time.process_time()-start,'warnings':[str(v.message) for v in w]}

def predict(fitted,texts):
    full,suffix,model=fitted;x=full.transform(texts)
    if suffix is not None:x=hstack([x,suffix.transform([tail(t) for t in texts])*.5]).tocsr()
    prob=model.predict_proba(x)
    return model.classes_[prob.argmax(axis=1)],prob.max(axis=1)

def choose_thresholds(pred,conf,gold):
    """Prespecified zero observed errors, >=8 samples, threshold >=.95.

    This is a conservative training-only heuristic, not a precision guarantee.
    None means defer that category. Never use external-fold labels here.
    """
    result={}
    for label in sorted(set(gold)):
        selected=(pred==label)&(conf>=.95)
        candidates=sorted(set(float(c) for c in conf[selected]))
        chosen=None;count=wrong=0
        if label!='harmful_content':
            for cutoff in candidates:
                keep=(pred==label)&(conf>=cutoff)
                if keep.sum()>=8 and not np.any(pred[keep]!=gold[keep]):
                    chosen=cutoff;count=int(keep.sum());wrong=0;break
        result[label]={'threshold':chosen,'inner_selected_count':count,'inner_wrong':wrong,
                       'eligible_count_at_095':int(selected.sum()),
                       'eligible_wrong_at_095':int(np.sum(pred[selected]!=gold[selected]))}
    return result

def classify_records(rows,pred,conf,by_id,scope,mode,thresholds=None,fold=None):
    out=records(rows,pred,conf,by_id,scope,mode)
    for r in out:
        if thresholds is not None:
            threshold=thresholds[r['prediction']]['threshold']
            r['gate_applies']=threshold is not None and r['confidence']>=threshold and r['prediction']!='harmful_content'
            r['gate_threshold']=threshold
        else:r['gate_threshold']=.95
        r['fold']=fold
    return out

def run(out):
    out=Path(out).resolve()
    if out==CACHE or not out.is_relative_to(CACHE):raise ValueError('Use a new directory inside experiments/.cache')
    if out.exists():raise FileExistsError('Never overwrite existing screen')
    start,cpu=time.perf_counter(),time.process_time()
    public=[];sources={};derived={}
    for name,rel in SOURCE_PATHS.items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('Input escapes repository')
        data=p.read_bytes();digest=sha(data)
        if digest!=EXPECTED_HASHES[name]:raise ValueError(f'Frozen dataset hash mismatch: {name}')
        rows=[json.loads(s) for s in data.decode().splitlines()]
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate input IDs')
        if not name.startswith('public_'):
            meta=json.loads((p.parent/'derivation.json').read_text())
            if meta.get('dataset_sha256')!=digest or meta.get('scope')!='public_paraphrase_input_diagnostic':
                raise ValueError('Unexpected transfer derivation provenance')
        sources[name]={'path':rel,'sha256':digest,'count':len(rows)}
        if name.startswith('public_'):public.extend([{**r,'_source_endpoint':name.removeprefix('public_')} for r in rows])
        else:derived[name]=rows
    by_id={r['id']:r for r in public};texts=[last(r) for r in public]
    y=np.array([r['labels']['category'] for r in public]);assert len(public)==400
    near=TfidfVectorizer(analyzer='char',ngram_range=(3,5)).fit_transform(texts)
    sim=(near@near.T).toarray();parents=list(range(len(public)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i in range(len(public)):
        for j in range(i):
            if sim[i,j]>=.82:parents[find(i)]=find(j)
    groups=np.array([find(i) for i in range(len(public))]);assert len(set(groups))==375
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=20261116).split(texts,y,groups))
    base_pred=np.empty(len(public),dtype=object);base_conf=np.zeros(len(public))
    predictions=[];threshold_audit=[];fit_audit=[]
    for outer,(train,test) in enumerate(folds):
        assert not set(groups[train])&set(groups[test])
        test_rows=[public[i] for i in test]
        for mode in ['raw','tail200']:
            fitted,meta=fit([texts[i] for i in train],y[train],mode=='tail200')
            pp,cc=predict(fitted,[texts[i] for i in test])
            predictions.extend(classify_records(test_rows,pp,cc,by_id,'public_oof',mode,fold=outer))
            fit_audit.append({'outer_fold':outer,'mode':mode,**meta})
            if mode=='raw':base_pred[test]=pp;base_conf[test]=cc
        inner_pred=np.empty(len(train),dtype=object);inner_conf=np.zeros(len(train));inner_manifest=[]
        innerfolds=StratifiedGroupKFold(4,shuffle=True,random_state=20261117+outer).split(train,y[train],groups[train])
        for inner,(itr,ite) in enumerate(innerfolds):
            tr,te=train[itr],train[ite]
            assert not set(groups[tr])&set(groups[te])
            assert not set(te)&set(test)
            fitted,meta=fit([texts[i] for i in tr],y[tr])
            pp,cc=predict(fitted,[texts[i] for i in te]);inner_pred[ite]=pp;inner_conf[ite]=cc
            inner_manifest.append({'fold':inner,'train_ids':[public[i]['id'] for i in tr],
                                   'validation_ids':[public[i]['id'] for i in te],**meta})
        thresholds=choose_thresholds(inner_pred,inner_conf,y[train])
        threshold_audit.append({'outer_fold':outer,'outer_test_ids':[public[i]['id'] for i in test],
                                'inner_folds':inner_manifest,'thresholds':thresholds})
        predictions.extend(classify_records(test_rows,base_pred[test],base_conf[test],by_id,'public_oof','nested_threshold',thresholds,outer))
        print('Completed outer fold',outer,flush=True)
    full_thresholds=choose_thresholds(base_pred,base_conf,y)
    for mode in ['raw','tail200']:
        fitted,meta=fit(texts,y,mode=='tail200');fit_audit.append({'mode':mode,'full_public':True,**meta})
        for scope,rows in {'public_in_sample':public,**derived}.items():
            pp,cc=predict(fitted,[last(r) for r in rows])
            predictions.extend(classify_records(rows,pp,cc,by_id,scope,mode))
            if mode=='raw':predictions.extend(classify_records(rows,pp,cc,by_id,scope,'nested_threshold',full_thresholds))
    results={mode:{scope:metrics([r for r in predictions if r['mode']==mode and r['scope']==scope])
              for scope in ['public_oof','public_in_sample',*derived]} for mode in ['raw','tail200','nested_threshold']}
    base={(r['scope'],r['id']):r for r in predictions if r['mode']=='raw'}
    changed=[];tradeoffs={}
    for mode in ['tail200','nested_threshold']:
        tradeoffs[mode]={}
        for scope in results[mode]:
            rs=[r for r in predictions if r['mode']==mode and r['scope']==scope and r['original_endpoint']=='tutor']
            tradeoffs[mode][scope]=dict(Counter(
                ('removed_correct' if b['correct'] else 'removed_wrong') if b['gate_applies'] and not r['gate_applies']
                else ('added_correct' if r['correct'] else 'added_wrong') if r['gate_applies'] and not b['gate_applies']
                else 'retained_changed_prediction' if r['gate_applies'] and b['prediction']!=r['prediction'] else 'other'
                for r in rs for b in [base[scope,r['id']]]))
            for r in rs:
                b=base[scope,r['id']]
                if r['prediction']!=b['prediction'] or r['gate_applies']!=b['gate_applies']:
                    changed.append({'scope':scope,'id':r['id'],'gold':r['gold'],'raw':b,'candidate':r})
    manifest={'scope':'offline_tutor_gate_cpu_development_not_holdout','git_head':git('rev-parse','HEAD').strip(),
              'sources':sources,'versions':{p:importlib.metadata.version(p) for p in ('scikit-learn','numpy','scipy','threadpoolctl')},
              'script_sha256':sha(Path(__file__).read_bytes()),'groups':375,'outer_folds':5,'group_threshold':.82,
              'outer_seed':20261116,'inner_folds':4,'inner_seed':'20261117 + outer_fold',
              'classifier':{'C':1000,'threshold':.95,'class_weight':'balanced','max_iter':500},
              'tail_feature':{'characters':200,'weight':.5,'only_if_length_exceeds':200,'last_user_only':True},
              'nested_threshold_rule':{'min_threshold':.95,'min_selected':8,'max_observed_errors':0,'no_valid_threshold':'defer'},
              'long_last_user_count':sum(len(t)>200 for t in texts),
              'thresholds_for_full_public_model':full_thresholds,'fit_audit':fit_audit,
              'total_wall_s':time.perf_counter()-start,'total_cpu_s':time.process_time()-cpu,
              'limitations':['Two fixed hypotheses on previously inspected development data; not pristine holdout.',
                             'Outer test labels are used only for evaluation; nested thresholds use only inner-OOF training-fold labels.',
                             'Vocabulary and IDF fit separately within every outer/inner train partition.',
                             'Derived rows are evaluation-only and share public sources; not independent generalization.',
                             'Classifier abstention hands off to existing LLM; fewer gate errors do not establish improved endpoint accuracy.',
                             'Class confidence is uncalibrated; zero observed inner errors and 8 examples are not a safety/statistical guarantee.',
                             'No guard runtime changes, model deployment, LLM/API calls or question-specific rules.']}
    exported_predictions=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions)
    exported_audit=json.dumps(threshold_audit,ensure_ascii=False,indent=2)+'\n'
    manifest['predictions_sha256']=sha(exported_predictions.encode())
    manifest['threshold_audit_sha256']=sha(exported_audit.encode())
    for source in sources.values():
        if sha((ROOT/source['path']).read_bytes())!=source['sha256']:raise ValueError('Input changed during diagnostic')
    out.mkdir(exist_ok=False)
    for name,data in [('manifest',manifest),('result',results),('threshold_audit',threshold_audit),('tutor_gate_tradeoffs',tradeoffs)]:dump(out/(name+'.json'),data)
    for name,data in [('predictions',predictions),('tutor_changes',changed)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    (out/'script.py').write_bytes(Path(__file__).read_bytes())
    summary=['# Tutor CPU gate screen','','Each cell: category correct / N; high-confidence applied / wrong.','','| Tutor subset | Raw | Tail200 | Nested class threshold |','|---|---:|---:|---:|']
    for scope in results['raw']:
        cells=[]
        for mode in results:
            r=results[mode][scope]['by_original_endpoint']['tutor']
            cells.append(f"{r['correct']}/{r['count']}; {r['high_confidence_nonharmful_count']}/{r['high_confidence_wrong']}")
        summary.append('| '+scope+' | '+' | '.join(cells)+' |')
    (out/'summary.md').write_text('\n'.join(summary)+'\n');print('\n'.join(summary));print('OUTPUT',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New local cache directory; existing results are never overwritten')
    with threadpool_limits(limits=1):run(parser.parse_args().output)
