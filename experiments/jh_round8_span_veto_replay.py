"""Offline high-confidence learned span veto; public training only, no API."""
import json,sys
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from pipelines.kimjaehak.span_tagger import SpanTagger,public_rows
from experiments.jh_guard_fusion_datasets import load_dataset
from experiments.jh_compare import score_rows


def main():
    dest=ROOT/'experiments/.cache/jh_r8_span_veto1';dest.mkdir();model=SpanTagger(public_rows());reports=[]
    for dataset,run in [('paraphrase','g06'),('novel','novel01'),('counterfactual',None),('relational',None)]:
        items,provenance=load_dataset(dataset,ROOT)
        predictions={r['id']:model.predict(r['candidate_answer']) for r in items if r.get('candidate_answer') and len(r['candidate_answer'])<=4000}
        if run:
            path=ROOT/'experiments/.cache'/('jh_guard_fusion_r8_'+run)/'r7_binary_guard/responses.jsonl';base=list(map(json.loads,path.read_text().splitlines()))
        for threshold in [.8,.9,.95]:
            selected={i:spans for i,(spans,c) in predictions.items() if c>=threshold and spans}
            if run:
                rows=deepcopy(base);changed=[]
                for row in rows:
                    response=row['response']
                    if row['id'] in selected and response['input']['decision']=='allow' and response.get('output',{}).get('grounded'):
                        response['output'].update(grounded=False,unsupported=selected[row['id']]);changed.append(row['id'])
                result=score_rows(items,rows,1,no_judge=True);lookup={r['id']:r for r in items};corrected=sum(not lookup[i]['labels']['grounded'] for i in changed)
                report={'dataset':dataset,'threshold':threshold,'changed':len(changed),'corrected':corrected,'regressed':len(changed)-corrected,'guard':result['guard'],'details':{k:result['guard_details'][k] for k in ['grounded_accuracy','span_f1']}}
            else:
                # No new live baseline: check whether a veto would damage known
                # supported controls; candidate detection is not a combined score.
                report={'dataset':dataset,'threshold':threshold,'standalone_flagged':len(selected),'flagged_supported':sum(r['labels']['grounded'] for r in items if r['id'] in selected),'flagged_unsupported':sum(not r['labels']['grounded'] for r in items if r['id'] in selected),'guard':None}
            reports.append(report);print(json.dumps(report),flush=True)
    (dest/'results.json').write_text(json.dumps({'scope':'public_trained_span_veto_frozen_B_or_control_diagnostic_not_holdout','api_calls':0,'results':reports},indent=2)+'\n');(dest/'COMPLETE').write_text('Span veto replay complete.\n')


if __name__=='__main__':main()
