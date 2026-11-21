"""Grouped public span confidence screen and frozen-B veto replay. API zero."""
import json,sys
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
from pipelines.kimjaehak.span_tagger import SpanTagger,public_rows,char_features,decode_mask
from experiments.jh_classifier_matrix import group_folds
from experiments.jh_compare import score_rows


def predict(model,text):
    p=model.model.predict_proba(model.vectorizer.transform(char_features(text)))
    mask=model.model.classes_[p.argmax(1)].astype(bool);confidence=p.max(1)
    return decode_mask(text,mask),{'minimum':float(confidence.min()),'quantile05':float(np.quantile(confidence,.05)),'quantile10':float(np.quantile(confidence,.1)),'positive_mean':float(confidence[mask].mean()) if mask.any() else 0.}


def main():
    dest=ROOT/'experiments/.cache/jh_r8_span_confidence1';dest.mkdir();train=public_rows();truth=np.array([not r['labels']['grounded'] for r in train]);_,folds=group_folds([r['candidate_answer'] for r in train],truth)
    oof={};allmodel=SpanTagger(train);out=[]
    for a,b in folds:
        model=SpanTagger([train[i] for i in a])
        for i in b:oof[int(i)]=predict(model,train[i]['candidate_answer'])
    for metric in ['minimum','quantile05','quantile10','positive_mean']:
        cv=[]
        for threshold in [.9,.95]:
            selected=np.array([bool(oof[i][0]) and oof[i][1][metric]>=threshold for i in range(len(train))])
            cv.append({'threshold':threshold,'false_alarm':int((selected&~truth).sum()),'detected':int((selected&truth).sum()),'missed':int((~selected&truth).sum())})
        replay=[]
        for run in ['g06','novel01']:
            path=ROOT/'experiments/.cache'/('jh_guard_fusion_r8_'+run);items=list(map(json.loads,(path/'inputs.jsonl').read_text().splitlines()));base=list(map(json.loads,(path/'r7_binary_guard/responses.jsonl').read_text().splitlines()));lookup={r['id']:r for r in items};pred={r['id']:predict(allmodel,r['candidate_answer']) for r in items if r.get('candidate_answer')}
            for threshold in [.9,.95]:
                rows=deepcopy(base);changed=[]
                for row in rows:
                    response=row['response'];v=pred.get(row['id'])
                    if v and v[0] and v[1][metric]>=threshold and response['input']['decision']=='allow' and response['output']['grounded']:
                        response['output'].update(grounded=False,unsupported=v[0]);changed.append(row['id'])
                scored=score_rows(items,rows,1,no_judge=True);correct=sum(not lookup[i]['labels']['grounded'] for i in changed)
                replay.append({'run':run,'threshold':threshold,'corrected':correct,'regressed':len(changed)-correct,'guard':scored['guard'],'span_f1':scored['guard_details']['span_f1']})
        report={'confidence':metric,'grouped_oof':cv,'frozen_B_replay':replay};out.append(report);print(json.dumps(report),flush=True)
    (dest/'results.json').write_text(json.dumps({'scope':'public86_grouped_span_confidence_and_frozen_B_replay_not_holdout','api_calls':0,'results':out},indent=2)+'\n');(dest/'COMPLETE').write_text('Span confidence screen complete.\n')


if __name__=='__main__':main()
