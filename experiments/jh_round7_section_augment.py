"""Offline KB-chunk augmentation of the scratch source classifier; no API calls."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from core.kb import load_chunks
from core.search import Searcher
from experiments.jh_section_ranker_cv import groups_for
rows=[r for r in map(json.loads,(ROOT/'public/tutor.jsonl').read_text().splitlines()) if len(r['labels'].get('evidence') or [])==1]
texts=[next(m['content'] for m in reversed(r['messages']) if m['role']=='user') for r in rows]
y=np.array([r['labels']['evidence'][0] for r in rows]);searcher=Searcher(load_chunks(ROOT/'kb'))
kb=[json.loads(s) for s in (ROOT/'experiments/.cache/jh_kb_probe_r2_base_off/derived_inputs.jsonl').read_text().splitlines()]
kt=[r['messages'][-1]['content'] for r in kb];ky=[r['labels']['evidence'][0] for r in kb]
folds=list(GroupKFold(5,shuffle=True,random_state=20261116).split(rows,groups=groups_for(rows)))
ids=searcher.section_ids();extra=[];labels=[]
for sid in ids:
 for chunk in searcher.chunks:
  if chunk.section_id==sid:
   # All KB text is supplied public material, not generated questions.
   extra.append(chunk.text);labels.append(sid)

def fit(train,weight):
 vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=1,max_features=60000,sublinear_tf=True)
 inputs=[texts[i] for i in train];targets=list(y[train]);weights=[1.]*len(inputs)
 if weight:inputs+=extra;targets+=labels;weights+=[weight]*len(extra)
 x=vectorizer.fit_transform(inputs)
 model=LogisticRegression(C=1000,max_iter=500,class_weight='balanced').fit(x,targets,sample_weight=np.array(weights))
 return vectorizer,model

def collect(v,m,x):
 p=m.predict_proba(v.transform(x));return m.classes_[p.argmax(1)],p.max(1)
def metrics(pred,conf,gold):
 correct=np.asarray(pred)==np.asarray(gold)
 return {'count':len(gold),'top1':int(correct.sum()),'gates':{str(t):{'accepted':int((conf>=t).sum()),'correct':int(correct[conf>=t].sum())} for t in [.5,.7,.9,.98]}}
reports=[]
with threadpool_limits(limits=1):
 for weight in [0,.05,.2,1.]:
  pred=np.empty(len(rows),dtype=object);conf=np.zeros(len(rows))
  for train,test in folds:
   v,m=fit(train,weight);pred[test],conf[test]=collect(v,m,[texts[i] for i in test])
  v,m=fit(np.arange(len(rows)),weight);kp,kc=collect(v,m,kt)
  result={'kb_chunk_weight':weight,'public_grouped':metrics(pred,conf,y),'kb55':metrics(kp,kc,ky)};reports.append(result);print(json.dumps(result),flush=True)
report={'scope':'public_grouped_source_classifier_and_prior_kb_development','api_calls':0,'kb55_used_to_train':False,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':reports}
(ROOT/'experiments/.cache/jh_r7_section_augmentation_repro.json').write_text(json.dumps(report,indent=2)+'\n')
