"""Offline screen of a narrowly scoped tutor normal/PII rescue gate."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from pipelines.kimjaehak.classifier_models import public_data


def main():
    root=ROOT/'experiments/.cache/jh_r7_hybrid_hybrid1'
    dest=ROOT/'experiments/.cache/jh_r7_tutor_screen1';dest.mkdir()
    manifest=json.loads((root/'manifest.json').read_text())
    rows,texts,vectors,_=public_data();labels=np.array([r['labels']['category'] for r in rows]);classes=np.unique(labels)
    if manifest['ids']!=[r['id'] for r in rows]:raise ValueError('Training order mismatch')
    p=np.zeros((len(rows),len(classes)))
    for fold in manifest['folds']:
        a,b=np.array(fold['train']),np.array(fold['test'])
        with threadpool_limits(limits=1):
            v=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=60000,sublinear_tf=True)
            m=LogisticRegression(C=1000,max_iter=500,class_weight='balanced')
            m.fit(v.fit_transform([texts[i] for i in a]),labels[a]);pred=m.predict_proba(v.transform([texts[i] for i in b]))
        for j,c in enumerate(m.classes_):p[b,np.flatnonzero(classes==c)[0]]=pred[:,j]
    pred=classes[p.argmax(1)];confidence=p.max(1);tutor=np.array([r['endpoint']=='tutor' for r in rows])
    eligible=tutor&np.isin(pred,['normal','pii_disclosure','out_of_material'])&(confidence<.95)
    results=[]
    for name,threshold in [('concat100',.9011610502863048),('blend_dense0.25',.8208260969823079)]:
        data=np.load(root/(name+'.npz'));hp=data['probabilities'];hc=data['classes'];hpred=hc[hp.argmax(1)];hconf=hp.max(1)
        accepted=eligible&np.isin(hpred,['normal','pii_disclosure'])&(hconf>=threshold)
        r={'model':name,'eligible':int(eligible.sum()),'new_direct':int(accepted.sum()),
            'new_direct_errors':int((hpred[accepted]!=labels[accepted]).sum()),
            'ids':[rows[i]['id'] for i in np.flatnonzero(accepted)]}
        results.append(r);print(json.dumps(r),flush=True)
    np.savez_compressed(dest/'baseline_lr_oof.npz',probabilities=p,classes=classes)
    (dest/'summary.json').write_text(json.dumps({'scope':'same_public_grouped_oof_tutor_rescue_screen_not_live_score',
        'api_calls':0,'tutor_count':int(tutor.sum()),'policy':'only uncertain original normal/pii/out-of-material; rescue normal/pii only',
        'results':results},indent=2)+'\n')


if __name__=='__main__':main()
